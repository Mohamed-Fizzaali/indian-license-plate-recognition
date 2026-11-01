package com.example.platecapture

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject
import java.io.File

internal enum class RecordState {
    PENDING,
    PROCESSING,
    COMPLETED,
    INVALID_FORMAT,
    FAILED,
    LEGACY_PROCESSED,
}

internal data class PlateRecord(
    val id: String,
    val originalPath: String,
    val cropPath: String? = null,
    val selection: NormalizedRect? = null,
    val offence: String? = null,
    val state: RecordState = RecordState.PENDING,
    val plateNumber: String? = null,
    val confidence: Double? = null,
    val validIndianFormat: Boolean? = null,
    val apiStatus: String? = null,
    val errorDetails: String? = null,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = createdAt,
    val submittedAt: Long? = null,
    val attemptGeneration: Long = 0,
)

internal object PendingRecordStore {
    private const val PREFERENCES = "plate_records_v2"
    private const val LEGACY_PREFERENCES = "record_status"
    private const val LEGACY_RESULT_MARKER = "legacy_processed_without_saved_result"
    private const val LEGACY_RESULT_MESSAGE = "Legacy processed record; the previous app version did not save its plate result."
    private const val INVALID_FORMAT_MESSAGE = "OCR returned text, but it does not pass the existing Indian-format check."
    private const val MISSING_FORMAT_MESSAGE = "The recognition response did not include a usable Indian-format status."
    private val originalName = Regex("^vehicle_\\d+(?:_\\d+)?\\.jpg$", RegexOption.IGNORE_CASE)
    private val mutableRecords = MutableStateFlow<List<PlateRecord>>(emptyList())
    val records: StateFlow<List<PlateRecord>> = mutableRecords.asStateFlow()

    @Synchronized
    fun initialize(context: Context) {
        val appContext = context.applicationContext
        val preferences = appContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val legacy = appContext.getSharedPreferences(LEGACY_PREFERENCES, Context.MODE_PRIVATE)
        val directory = File(appContext.filesDir, "captures")
        val editor = preferences.edit()
        var changed = false

        directory.listFiles { file ->
            file.isFile && originalName.matches(file.name) && ImageFiles.hasReadableImage(file.absolutePath)
        }.orEmpty().forEach { file ->
            val id = file.nameWithoutExtension
            if (!preferences.contains(id)) {
                val legacyStatus = legacy.getString(file.name, "PENDING")
                val migrated = if (legacyStatus == "PROCESSED") {
                    PlateRecord(
                        id = id,
                        originalPath = file.absolutePath,
                        state = RecordState.LEGACY_PROCESSED,
                        apiStatus = LEGACY_RESULT_MARKER,
                        errorDetails = LEGACY_RESULT_MESSAGE,
                        createdAt = file.lastModified(),
                    )
                } else {
                    PlateRecord(id = id, originalPath = file.absolutePath, createdAt = file.lastModified())
                }
                editor.putString(id, encode(migrated))
                changed = true
            }
        }

        preferences.all.values.filterIsInstance<String>()
            .mapNotNull(::decode)
            .filter {
                it.state == RecordState.COMPLETED &&
                    it.apiStatus.equals("success", ignoreCase = true) &&
                    it.plateNumber.isNullOrBlank()
            }
            .forEach { record ->
                editor.putString(
                    record.id,
                    encode(
                        record.copy(
                            state = RecordState.LEGACY_PROCESSED,
                            apiStatus = LEGACY_RESULT_MARKER,
                            errorDetails = LEGACY_RESULT_MESSAGE,
                        ),
                    ),
                )
                changed = true
            }

        preferences.all.values.filterIsInstance<String>()
            .mapNotNull(::decode)
            .filter {
                it.state == RecordState.COMPLETED &&
                    it.apiStatus.equals("success", ignoreCase = true) &&
                    !it.plateNumber.isNullOrBlank() &&
                    it.validIndianFormat == false
            }
            .forEach { record ->
                editor.putString(
                    record.id,
                    encode(record.copy(
                        state = RecordState.INVALID_FORMAT,
                        errorDetails = record.errorDetails ?: INVALID_FORMAT_MESSAGE,
                    )),
                )
                changed = true
            }

        preferences.all.values.filterIsInstance<String>()
            .mapNotNull(::decode)
            .filter {
                it.state == RecordState.COMPLETED &&
                    it.apiStatus.equals("success", ignoreCase = true) &&
                    !it.plateNumber.isNullOrBlank() &&
                    it.validIndianFormat == null
            }
            .forEach { record ->
                editor.putString(
                    record.id,
                    encode(record.copy(
                        state = RecordState.FAILED,
                        errorDetails = record.errorDetails ?: MISSING_FORMAT_MESSAGE,
                    )),
                )
                changed = true
            }

        if (changed) check(editor.commit()) { "Could not migrate saved records" }
        publish(preferences)
    }

    @Synchronized
    fun get(context: Context, id: String): PlateRecord? = decode(
        context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(id, null),
    )

    @Synchronized
    fun save(context: Context, record: PlateRecord) {
        val preferences = context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        check(preferences.edit().putString(record.id, encode(record)).commit()) {
            "Could not save record"
        }
        publish(preferences)
    }

    @Synchronized
    fun updateAttempt(
        context: Context,
        id: String,
        generation: Long,
        update: (PlateRecord) -> PlateRecord,
    ): Boolean {
        val preferences = context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val current = decode(preferences.getString(id, null)) ?: return false
        if (current.state != RecordState.PROCESSING || current.attemptGeneration != generation) return false
        val updated = update(current)
        check(updated.id == id && updated.attemptGeneration == generation) {
            "Attempt update changed its record identity"
        }
        check(preferences.edit().putString(id, encode(updated)).commit()) {
            "Could not save recognition result"
        }
        publish(preferences)
        return true
    }

    @Synchronized
    fun beginRetry(context: Context, id: String): PlateRecord? {
        val preferences = context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val current = decode(preferences.getString(id, null))
            ?.takeIf { it.state == RecordState.FAILED || it.state == RecordState.INVALID_FORMAT }
            ?: return null
        val retrying = current.copy(
            state = RecordState.PROCESSING,
            plateNumber = null,
            confidence = null,
            validIndianFormat = null,
            apiStatus = null,
            errorDetails = null,
            updatedAt = System.currentTimeMillis(),
            attemptGeneration = current.attemptGeneration + 1,
        )
        check(preferences.edit().putString(id, encode(retrying)).commit()) {
            "Could not save retry status"
        }
        publish(preferences)
        return retrying
    }

    private fun publish(preferences: android.content.SharedPreferences) {
        mutableRecords.value = preferences.all.values
            .filterIsInstance<String>()
            .mapNotNull(::decode)
            .sortedByDescending { it.createdAt }
    }

    private fun encode(record: PlateRecord): String = JSONObject().apply {
        put("id", record.id)
        put("original_path", record.originalPath)
        put("crop_path", record.cropPath ?: JSONObject.NULL)
        put("selection", record.selection?.let { rect ->
            JSONObject().put("left", rect.left).put("top", rect.top)
                .put("right", rect.right).put("bottom", rect.bottom)
        } ?: JSONObject.NULL)
        put("offence", record.offence ?: JSONObject.NULL)
        put("state", record.state.name)
        put("plate_number", record.plateNumber ?: JSONObject.NULL)
        put("confidence", record.confidence ?: JSONObject.NULL)
        put("valid_indian_format", record.validIndianFormat ?: JSONObject.NULL)
        put("api_status", record.apiStatus ?: JSONObject.NULL)
        put("error_details", record.errorDetails ?: JSONObject.NULL)
        put("created_at", record.createdAt)
        put("updated_at", record.updatedAt)
        put("submitted_at", record.submittedAt ?: JSONObject.NULL)
        put("attempt_generation", record.attemptGeneration)
    }.toString()

    private fun decode(encoded: String?): PlateRecord? = runCatching {
        val json = JSONObject(encoded ?: return null)
        val id = json.getString("id")
        val state = RecordState.valueOf(json.getString("state"))
        PlateRecord(
            id = id,
            originalPath = json.getString("original_path"),
            cropPath = json.optNullableString("crop_path"),
            selection = (json.opt("selection") as? JSONObject)?.let {
                NormalizedRect(
                    it.getDouble("left").toFloat(),
                    it.getDouble("top").toFloat(),
                    it.getDouble("right").toFloat(),
                    it.getDouble("bottom").toFloat(),
                )
            },
            offence = json.optNullableString("offence"),
            state = state,
            plateNumber = json.optNullableString("plate_number"),
            confidence = json.opt("confidence").let { value -> (value as? Number)?.toDouble() },
            validIndianFormat = json.opt("valid_indian_format") as? Boolean,
            apiStatus = json.optNullableString("api_status"),
            errorDetails = json.optNullableString("error_details"),
            createdAt = json.optLong("created_at", System.currentTimeMillis()),
            updatedAt = json.optLong("updated_at", System.currentTimeMillis()),
            submittedAt = json.opt("submitted_at").let { value -> (value as? Number)?.toLong() },
            attemptGeneration = json.optLong("attempt_generation", 0L),
        )
    }.getOrNull()

    private fun JSONObject.optNullableString(key: String): String? = opt(key)
        .takeUnless { it == null || it == JSONObject.NULL }
        ?.toString()
}
