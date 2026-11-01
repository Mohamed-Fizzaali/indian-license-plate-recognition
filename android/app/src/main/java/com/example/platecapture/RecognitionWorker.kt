package com.example.platecapture

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.Operation
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

internal object RecognitionWork {
    private const val INPUT_RECORD_ID = "record_id"
    private const val INPUT_ATTEMPT_GENERATION = "attempt_generation"
    private const val ATTEMPT_TAG_PREFIX = "recognition_attempt_"
    private const val TAG = "RecognitionWork"

    suspend fun enqueue(
        context: Context,
        recordId: String,
        attemptGeneration: Long,
        policy: ExistingWorkPolicy = ExistingWorkPolicy.KEEP,
    ) {
        withContext(Dispatchers.IO) {
            enqueueOperation(context, recordId, attemptGeneration, policy).result.get()
        }
    }

    suspend fun reconcile(context: Context, records: List<PlateRecord>) = withContext(Dispatchers.IO) {
        val manager = WorkManager.getInstance(context.applicationContext)
        records.filter { it.state == RecordState.PROCESSING }.forEach { record ->
            val infos = runCatching { manager.getWorkInfosForUniqueWork(workName(record.id)).get() }
                .onFailure { Log.w(TAG, "Could not inspect work for ${record.id}", it) }
                .getOrNull()
            val activeWork = infos.orEmpty().filter {
                it.state == WorkInfo.State.ENQUEUED ||
                    it.state == WorkInfo.State.BLOCKED ||
                    it.state == WorkInfo.State.RUNNING
            }
            val hasCurrentAttempt = activeWork.any {
                it.tags.contains(attemptTag(record.attemptGeneration)) ||
                    (record.attemptGeneration == 0L && it.tags.none { tag -> tag.startsWith(ATTEMPT_TAG_PREFIX) })
            }
            if (hasCurrentAttempt) return@forEach

            runCatching {
                enqueueOperation(
                    context,
                    record.id,
                    record.attemptGeneration,
                    policy = ExistingWorkPolicy.APPEND_OR_REPLACE,
                )
                    .result.get()
            }.onFailure { error ->
                runCatching {
                    PendingRecordStore.updateAttempt(context, record.id, record.attemptGeneration) {
                        it.copy(
                            state = RecordState.FAILED,
                            apiStatus = "enqueue_error",
                            errorDetails = error.message ?: "Could not schedule recognition",
                            updatedAt = System.currentTimeMillis(),
                        )
                    }
                }
                Log.e(TAG, "Could not re-enqueue work for ${record.id}", error)
            }
        }
    }

    private fun enqueueOperation(
        context: Context,
        recordId: String,
        attemptGeneration: Long,
        policy: ExistingWorkPolicy,
    ): Operation {
        val request = OneTimeWorkRequestBuilder<RecognitionWorker>()
            .setInputData(
                Data.Builder()
                    .putString(INPUT_RECORD_ID, recordId)
                    .putLong(INPUT_ATTEMPT_GENERATION, attemptGeneration)
                    .build(),
            )
            .addTag(attemptTag(attemptGeneration))
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build(),
            )
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()
        return WorkManager.getInstance(context.applicationContext)
            .enqueueUniqueWork(
                workName(recordId),
                policy,
                request,
            )
    }

    private fun workName(recordId: String) = "recognition_$recordId"
    private fun attemptTag(generation: Long) = "$ATTEMPT_TAG_PREFIX$generation"
}

internal class RecognitionWorker(
    context: Context,
    parameters: WorkerParameters,
) : CoroutineWorker(context, parameters) {
    override suspend fun doWork(): Result {
        val recordId = inputData.getString("record_id") ?: return Result.failure()
        val generation = inputData.getLong("attempt_generation", 0L)
        val record = try {
            PendingRecordStore.get(applicationContext, recordId)
        } catch (error: Exception) {
            return finish(
                recordId,
                generation,
                RecordState.FAILED,
                "client_error",
                error.message ?: "Could not load the saved recognition record",
                result = Result.failure(),
            )
        } ?: return Result.failure()

        if (record.state != RecordState.PROCESSING || record.attemptGeneration != generation) {
            return Result.success()
        }

        return try {
            val crop = record.cropPath?.let(::File)
            if (crop?.isFile != true) {
                return finish(
                    recordId,
                    generation,
                    RecordState.FAILED,
                    "client_error",
                    "Selected plate crop is unavailable",
                    result = Result.failure(),
                )
            }

            val response = withContext(Dispatchers.IO) { RecognitionApi.recognize(crop) }
            handleResponse(recordId, generation, response)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: RecognitionHttpException) {
            handleRequestFailure(
                recordId,
                generation,
                error,
                error.statusCode in setOf(408, 429) || error.statusCode >= 500,
            )
        } catch (error: RecognitionResponseException) {
            finish(
                recordId,
                generation,
                RecordState.FAILED,
                "invalid_response",
                error.message ?: "Recognition server returned an invalid response",
                result = Result.failure(),
            )
        } catch (error: IOException) {
            handleRequestFailure(recordId, generation, error, retryable = true)
        } catch (error: Exception) {
            finish(
                recordId,
                generation,
                RecordState.FAILED,
                "client_error",
                error.message ?: "Recognition request failed",
                result = Result.failure(),
            )
        }
    }

    private suspend fun handleResponse(
        recordId: String,
        generation: Long,
        response: RecognitionResponse,
    ): Result {
        val status = response.status?.trim()?.lowercase()
        response.protocolError?.let { detail ->
            return finish(
                recordId,
                generation,
                RecordState.FAILED,
                response.status ?: "invalid_response",
                response.error ?: detail,
                response,
                Result.failure(),
            )
        }

        return when (status) {
            "success" -> when {
                response.plateNumber.isNullOrBlank() -> finish(
                    recordId,
                    generation,
                    RecordState.FAILED,
                    response.status,
                    response.error ?: "The API reported success without returning plate text.",
                    response,
                    Result.failure(),
                )
                response.validIndianFormat == null -> finish(
                    recordId,
                    generation,
                    RecordState.FAILED,
                    response.status,
                    response.error ?: "The API reported success but omitted a valid_indian_format result.",
                    response,
                    Result.failure(),
                )
                response.validIndianFormat -> finish(
                    recordId,
                    generation,
                    RecordState.COMPLETED,
                    response.status,
                    null,
                    response,
                    Result.success(),
                )
                else -> finish(
                    recordId,
                    generation,
                    RecordState.INVALID_FORMAT,
                    response.status,
                    "OCR returned text, but it does not pass the existing Indian-format check.",
                    response,
                    Result.success(),
                )
            }
            "plate_not_found", "recognition_failed" -> {
                val detail = if (status == "plate_not_found") {
                    "No license plate was found in the selected crop."
                } else {
                    "The recognition service could not read a plate number from the selected crop."
                }
                finish(
                    recordId,
                    generation,
                    RecordState.FAILED,
                    response.status,
                    response.error ?: detail,
                    response,
                    Result.failure(),
                )
            }
            else -> finish(
                recordId,
                generation,
                RecordState.FAILED,
                response.status ?: "invalid_response",
                response.error ?: when (status) {
                    "error" -> "Recognition service reported an error"
                    null -> "Recognition response did not include a status"
                    else -> "Unrecognized API status: ${response.status}"
                },
                response,
                Result.failure(),
            )
        }
    }

    private suspend fun handleRequestFailure(
        recordId: String,
        generation: Long,
        error: IOException,
        retryable: Boolean,
    ): Result {
        if (retryable && runAttemptCount < MAX_RETRIES) return Result.retry()
        val status = (error as? RecognitionHttpException)?.let { "http_${it.statusCode}" } ?: "transport_error"
        return finish(
            recordId,
            generation,
            RecordState.FAILED,
            status,
            error.message ?: "Could not contact recognition service",
            result = Result.failure(),
        )
    }

    private suspend fun finish(
        recordId: String,
        generation: Long,
        state: RecordState,
        apiStatus: String?,
        detail: String?,
        response: RecognitionResponse? = null,
        result: Result,
    ): Result {
        return try {
            val updated = withContext(Dispatchers.IO) {
                PendingRecordStore.updateAttempt(applicationContext, recordId, generation) { current ->
                    current.copy(
                        state = state,
                        plateNumber = if (response == null) current.plateNumber else response.plateNumber,
                        confidence = if (response == null) current.confidence else response.confidence,
                        validIndianFormat = if (response == null) current.validIndianFormat else response.validIndianFormat,
                        apiStatus = apiStatus,
                        errorDetails = detail,
                        updatedAt = System.currentTimeMillis(),
                    )
                }
            }
            if (updated) result else Result.success()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.e(TAG, "Could not save result for $recordId", error)
            if (runAttemptCount < MAX_RETRIES) Result.retry() else Result.failure()
        }
    }

    private companion object {
        const val MAX_RETRIES = 3
        const val TAG = "RecognitionWorker"
    }
}
