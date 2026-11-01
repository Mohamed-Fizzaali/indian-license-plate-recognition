package com.example.platecapture

import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

internal data class RecognitionResponse(
    val plateNumber: String?,
    val confidence: Double?,
    val validIndianFormat: Boolean?,
    val status: String?,
    val error: String? = null,
    val protocolError: String? = null,
)

internal class RecognitionHttpException(
    val statusCode: Int,
    message: String,
) : IOException(message)

internal class RecognitionResponseException(message: String, cause: Throwable? = null) :
    IOException(message, cause)

internal object RecognitionApi {
    private const val ENDPOINT = "http://192.168.137.1:5000/recognize"

    fun recognize(crop: File): RecognitionResponse {
        if (!crop.isFile) throw IOException("Selected plate crop is unavailable")

        val boundary = "----PlateCapture${UUID.randomUUID().toString().replace("-", "")}"
        val connection = URL(ENDPOINT).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 15_000
        connection.readTimeout = 45_000
        connection.useCaches = false
        connection.doOutput = true
        connection.setRequestProperty("Accept", "application/json")
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")

        try {
            connection.outputStream.use { output ->
                output.write("--$boundary\r\n".toByteArray(Charsets.UTF_8))
                output.write(
                    "Content-Disposition: form-data; name=\"image\"; filename=\"plate.jpg\"\r\n".toByteArray(Charsets.UTF_8),
                )
                output.write("Content-Type: image/jpeg\r\n\r\n".toByteArray(Charsets.UTF_8))
                crop.inputStream().use { it.copyTo(output) }
                output.write("\r\n--$boundary--\r\n".toByteArray(Charsets.UTF_8))
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val message = runCatching { JSONObject(body).optString("error") }
                    .getOrNull()
                    ?.takeIf { it.isNotBlank() }
                throw RecognitionHttpException(code, message ?: "Recognition request failed (HTTP $code)")
            }

            return parseResponse(body)
        } catch (error: JSONException) {
            throw RecognitionResponseException("Recognition server returned invalid JSON", error)
        } finally {
            connection.disconnect()
        }
    }

    private fun parseResponse(body: String): RecognitionResponse {
        val json = JSONObject(body)
        val rawPlateNumber = json.opt("plate_number")
        val plateNumber = (rawPlateNumber as? String)?.takeIf { it.isNotBlank() }
        val rawConfidence = json.opt("confidence")
        val confidence = when {
            rawConfidence is Number -> rawConfidence.toDouble()
            rawConfidence is String -> rawConfidence.toDoubleOrNull()
            else -> null
        }?.takeIf { it.isFinite() }
        val rawFormat = json.opt("valid_indian_format")
        val validIndianFormat = when {
            rawFormat is Boolean -> rawFormat
            rawFormat is String -> rawFormat.toBooleanStrictOrNull()
            else -> null
        }
        val rawStatus = json.opt("status")
        val status = (rawStatus as? String)?.takeIf { it.isNotBlank() }
        val error = (json.opt("error") as? String)?.takeIf { it.isNotBlank() }
        val protocolErrors = buildList {
            if (status == null) add("Recognition response is missing a valid status")
            if (rawPlateNumber != null && rawPlateNumber != JSONObject.NULL && rawPlateNumber !is String) {
                add("Recognition response plate_number is not text")
            }
            if (json.has("valid_indian_format") && rawFormat != JSONObject.NULL && validIndianFormat == null) {
                add("Recognition response valid_indian_format is invalid")
            }
        }

        return RecognitionResponse(
            plateNumber = plateNumber,
            confidence = confidence,
            validIndianFormat = validIndianFormat,
            status = status,
            error = error,
            protocolError = protocolErrors.takeIf { it.isNotEmpty() }?.joinToString("; "),
        )
    }
}
