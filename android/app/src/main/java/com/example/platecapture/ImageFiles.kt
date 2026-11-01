package com.example.platecapture

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max

internal data class NormalizedRect(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)

internal object ImageFiles {
    fun hasReadableImage(path: String): Boolean = runCatching {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        bounds.outWidth > 0 && bounds.outHeight > 0
    }.getOrDefault(false)

    fun decodePreview(path: String, targetWidth: Int, targetHeight: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Could not decode captured photo" }

        val orientation = ExifInterface(path).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
        val (displayWidth, displayHeight) = displaySize(bounds.outWidth, bounds.outHeight, orientation)
        var sample = 1
        while (displayWidth / sample > max(1, targetWidth) * 1.5 ||
            displayHeight / sample > max(1, targetHeight) * 1.5
        ) {
            sample *= 2
        }

        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeFile(path, options)
            ?: error("Could not load captured photo")
        return orient(bitmap, orientation)
    }

    fun writeSelectionCrop(sourcePath: String, selection: NormalizedRect, output: File): File {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(sourcePath, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Could not read original photo dimensions" }

        val orientation = ExifInterface(sourcePath).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
        val coordinates = listOf(
            selection.left to selection.top,
            selection.right to selection.top,
            selection.left to selection.bottom,
            selection.right to selection.bottom,
        ).map { displayToRaw(it.first, it.second, orientation) }
        val rawLeft = coordinates.minOf { it.first }
        val rawTop = coordinates.minOf { it.second }
        val rawRight = coordinates.maxOf { it.first }
        val rawBottom = coordinates.maxOf { it.second }
        val left = floor(rawLeft * bounds.outWidth).toInt().coerceIn(0, bounds.outWidth - 1)
        val top = floor(rawTop * bounds.outHeight).toInt().coerceIn(0, bounds.outHeight - 1)
        val right = ceil(rawRight * bounds.outWidth).toInt().coerceIn(left + 1, bounds.outWidth)
        val bottom = ceil(rawBottom * bounds.outHeight).toInt().coerceIn(top + 1, bounds.outHeight)
        val region = Rect(left, top, right, bottom)

        var sample = 1
        while (max(region.width(), region.height()) / sample > 1600) sample *= 2
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val stream = FileInputStream(sourcePath)
        val decoder = try {
            android.graphics.BitmapRegionDecoder.newInstance(stream, false)
                ?: error("Could not open photo decoder")
        } catch (error: Exception) {
            stream.close()
            throw error
        }

        val cropped = try {
            decoder.decodeRegion(region, options) ?: error("Could not crop selected region")
        } finally {
            decoder.recycle()
            stream.close()
        }
        val oriented = orient(cropped, orientation)
        output.parentFile?.mkdirs()
        try {
            FileOutputStream(output).use { file ->
                check(oriented.compress(Bitmap.CompressFormat.JPEG, 95, file)) {
                    "Could not save selected region"
                }
            }
        } finally {
            if (oriented !== cropped) oriented.recycle()
            cropped.recycle()
        }
        return output
    }

    private fun displaySize(width: Int, height: Int, orientation: Int): Pair<Int, Int> =
        if (orientation in 5..8) height to width else width to height

    private fun displayToRaw(x: Float, y: Float, orientation: Int): Pair<Float, Float> = when (orientation) {
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> 1f - x to y
        ExifInterface.ORIENTATION_ROTATE_180 -> 1f - x to 1f - y
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> x to 1f - y
        ExifInterface.ORIENTATION_TRANSPOSE -> y to x
        ExifInterface.ORIENTATION_ROTATE_90 -> y to 1f - x
        ExifInterface.ORIENTATION_TRANSVERSE -> 1f - y to 1f - x
        ExifInterface.ORIENTATION_ROTATE_270 -> 1f - y to x
        else -> x to y
    }

    private fun orient(bitmap: Bitmap, orientation: Int): Bitmap {
        if (orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED
        ) return bitmap

        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()
        val values = when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> floatArrayOf(-1f, 0f, width, 0f, 1f, 0f, 0f, 0f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> floatArrayOf(-1f, 0f, width, 0f, -1f, height, 0f, 0f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> floatArrayOf(1f, 0f, 0f, 0f, -1f, height, 0f, 0f, 1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> floatArrayOf(0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)
            ExifInterface.ORIENTATION_ROTATE_90 -> floatArrayOf(0f, -1f, height, 1f, 0f, 0f, 0f, 0f, 1f)
            ExifInterface.ORIENTATION_TRANSVERSE -> floatArrayOf(0f, -1f, height, -1f, 0f, width, 0f, 0f, 1f)
            ExifInterface.ORIENTATION_ROTATE_270 -> floatArrayOf(0f, 1f, 0f, -1f, 0f, width, 0f, 0f, 1f)
            else -> return bitmap
        }
        val matrix = Matrix().apply { setValues(values) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
