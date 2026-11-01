package com.example.platecapture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.Surface
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items as columnItems
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.Saver
import androidx.compose.runtime.saveable.listSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.math.min

private val Ink = Color(0xFF101820)
private val Paper = Color(0xFFF6F4EF)
private val Accent = Color(0xFF42D6C4)
private val selectionSaver: Saver<NormalizedRect?, Any> = listSaver(
    save = { selection ->
        selection?.let { listOf(it.left, it.top, it.right, it.bottom) } ?: emptyList()
    },
    restore = { values ->
        values.takeIf { it.size == 4 }?.let {
            NormalizedRect(it[0], it[1], it[2], it[3])
        }
    },
)
private enum class RecordPage { PENDING, HISTORY }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { PlateCaptureApp() } }
    }
}

@Composable
private fun PlateCaptureApp() {
    val context = LocalContext.current
    val activity = context as? Activity
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    var cameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    var permissionRequested by rememberSaveable { mutableStateOf(false) }
    var photoPath by rememberSaveable { mutableStateOf<String?>(null) }
    var showRecords by rememberSaveable { mutableStateOf(false) }
    var recordPage by rememberSaveable { mutableStateOf(RecordPage.PENDING) }
    var reviewFromRecords by rememberSaveable { mutableStateOf(false) }
    var isSubmitting by remember { mutableStateOf(false) }
    var captureRecordError by remember { mutableStateOf<String?>(null) }
    var reviewError by rememberSaveable { mutableStateOf<String?>(null) }
    val records by PendingRecordStore.records.collectAsState()
    val pendingRecords = records.filter { it.state == RecordState.PENDING }
    val historyRecords = records.filter { it.state != RecordState.PENDING }.sortedByDescending { it.submittedAt ?: it.createdAt }
    var selectedRegion by rememberSaveable(stateSaver = selectionSaver) {
        mutableStateOf<NormalizedRect?>(null)
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraPermission = granted
        permissionRequested = true
    }

    LaunchedEffect(context) {
        withContext(Dispatchers.IO) { PendingRecordStore.initialize(context) }
        withContext(Dispatchers.IO) {
            RecognitionWork.reconcile(context, PendingRecordStore.records.value)
        }
    }

    LaunchedEffect(cameraPermission, permissionRequested) {
        if (!cameraPermission && !permissionRequested) {
            permissionRequested = true
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    fun leaveReview() {
        photoPath = null
        selectedRegion = null
        reviewError = null
        showRecords = reviewFromRecords
        reviewFromRecords = false
    }

    fun retryRecord(record: PlateRecord) {
        if (record.state != RecordState.FAILED && record.state != RecordState.INVALID_FORMAT) return
        scope.launch {
            val retrying = withContext(Dispatchers.IO) {
                PendingRecordStore.beginRetry(context, record.id)
            } ?: return@launch
            try {
                RecognitionWork.enqueue(
                    context,
                    record.id,
                    retrying.attemptGeneration,
                    policy = androidx.work.ExistingWorkPolicy.APPEND_OR_REPLACE,
                )
            } catch (error: Exception) {
                withContext(Dispatchers.IO) {
                    PendingRecordStore.updateAttempt(context, record.id, retrying.attemptGeneration) {
                        it.copy(
                            state = RecordState.FAILED,
                            apiStatus = "enqueue_error",
                            errorDetails = error.message ?: "Could not schedule recognition retry",
                            updatedAt = System.currentTimeMillis(),
                        )
                    }
                }
            }
        }
    }

    BackHandler(enabled = showRecords || photoPath != null) {
        when {
            photoPath != null && !isSubmitting -> leaveReview()
            photoPath == null && showRecords -> showRecords = false
        }
    }

    DisposableEffect(lifecycleOwner, context) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                cameraPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    when {
        !cameraPermission -> PermissionScreen(
            askedBefore = permissionRequested,
            showRationale = activity?.shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) == true,
            onRequest = { permissionLauncher.launch(Manifest.permission.CAMERA) },
            onSettings = { openApplicationSettings(context) },
        )
        photoPath == null && showRecords -> RecordsScreen(
            pending = pendingRecords,
            history = historyRecords,
            selectedPage = recordPage,
            onPageSelected = { recordPage = it },
            onCamera = { showRecords = false },
            onSelect = { record ->
                photoPath = record.originalPath
                reviewFromRecords = true
                showRecords = false
                reviewError = null
                selectedRegion = null
            },
            onRetry = ::retryRecord,
        )
        photoPath == null -> CameraScreen(
            pendingCount = pendingRecords.size,
            recordError = captureRecordError,
            onOpenRecords = {
                recordPage = RecordPage.PENDING
                showRecords = true
            },
            onCaptured = { path ->
                captureRecordError = null
                scope.launch {
                    try {
                        withContext(Dispatchers.IO) {
                            val file = File(path)
                            PendingRecordStore.save(
                                context,
                                PlateRecord(
                                    id = file.nameWithoutExtension,
                                    originalPath = file.absolutePath,
                                    createdAt = file.lastModified(),
                                ),
                            )
                        }
                    } catch (error: Exception) {
                        captureRecordError = error.message ?: "Could not save pending record status"
                    }
                }
                reviewFromRecords = false
                reviewError = null
                selectedRegion = null
            },
        )
        else -> ReviewScreen(
            photoPath = photoPath!!,
            selection = selectedRegion,
            onSelectionChanged = {
                selectedRegion = it
                reviewError = null
            },
            onClear = {
                selectedRegion = null
                reviewError = null
            },
            error = reviewError,
            submitting = isSubmitting,
            onConfirm = { offence ->
                if (isSubmitting) return@ReviewScreen
                val source = photoPath ?: return@ReviewScreen
                val region = selectedRegion?.takeIf { it.isUsable() } ?: return@ReviewScreen
                isSubmitting = true
                reviewError = null
                scope.launch {
                    try {
                        val record = withContext(Dispatchers.IO) {
                            val sourceFile = File(source)
                            val id = sourceFile.nameWithoutExtension
                            val previous = PendingRecordStore.get(context, id)
                            check(previous == null || previous.state == RecordState.PENDING) {
                                "This record has already been submitted"
                            }
                            val crop = ImageFiles.writeSelectionCrop(
                                source,
                                region,
                                File(context.filesDir, "captures/${id}_plate.jpg"),
                            )
                            PlateRecord(
                                id = id,
                                originalPath = sourceFile.absolutePath,
                                cropPath = crop.absolutePath,
                                selection = region,
                                offence = offence,
                                state = RecordState.PROCESSING,
                                createdAt = previous?.createdAt ?: sourceFile.lastModified(),
                                updatedAt = System.currentTimeMillis(),
                                submittedAt = previous?.submittedAt ?: System.currentTimeMillis(),
                                attemptGeneration = (previous?.attemptGeneration ?: 0L) + 1L,
                            ).also { PendingRecordStore.save(context, it) }
                        }
                        try {
                            RecognitionWork.enqueue(
                                context,
                                record.id,
                                record.attemptGeneration,
                                policy = androidx.work.ExistingWorkPolicy.APPEND_OR_REPLACE,
                            )
                        } catch (error: Exception) {
                            withContext(Dispatchers.IO) {
                                PendingRecordStore.updateAttempt(context, record.id, record.attemptGeneration) {
                                    it.copy(
                                        state = RecordState.FAILED,
                                        apiStatus = "enqueue_error",
                                        errorDetails = error.message ?: "Could not schedule recognition",
                                        updatedAt = System.currentTimeMillis(),
                                    )
                                }
                            }
                        }
                        photoPath = null
                        selectedRegion = null
                        reviewFromRecords = false
                        showRecords = true
                        recordPage = RecordPage.PENDING
                    } catch (error: Exception) {
                        reviewError = error.message ?: "Could not submit this record"
                    } finally {
                        isSubmitting = false
                    }
                }
            },
            onCancel = { if (!isSubmitting) leaveReview() },
        )
    }
}

@Composable
private fun PermissionScreen(
    askedBefore: Boolean,
    showRationale: Boolean,
    onRequest: () -> Unit,
    onSettings: () -> Unit,
) {
    val blocked = askedBefore && !showRationale
    Column(
        modifier = Modifier.fillMaxSize().background(Ink).padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Camera access is needed to capture a vehicle photo.", color = Paper, fontSize = 20.sp, textAlign = TextAlign.Center)
        Spacer(Modifier.height(12.dp))
        Text(
            if (blocked) "Enable camera access in app settings to continue."
            else "The app only uses the camera for a photo you choose to capture.",
            color = Paper.copy(alpha = 0.72f),
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = if (blocked) onSettings else onRequest) {
            Text(if (blocked) "Open settings" else "Allow camera")
        }
    }
}

@Composable
private fun CameraScreen(
    pendingCount: Int,
    recordError: String?,
    onOpenRecords: () -> Unit,
    onCaptured: (String) -> Unit,
) {
    val context = LocalContext.current
    val view = LocalView.current
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var isCapturing by remember { mutableStateOf(false) }
    var cameraAttempt by rememberSaveable { mutableStateOf(0) }

    Column(Modifier.fillMaxSize().background(Ink)) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                "PLATE CAPTURE",
                color = Paper,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp,
            )
            TextButton(onClick = onOpenRecords) {
                Text("Records ($pendingCount)")
            }
        }
        Box(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 10.dp)) {
            key(cameraAttempt) {
                CameraPreview(
                    modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(22.dp)),
                    onCaptureReady = { imageCapture = it },
                    onError = { cameraError = it },
                )
            }
            if (cameraError != null) {
                Column(
                    Modifier.align(Alignment.Center).padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(cameraError.orEmpty(), color = Paper, textAlign = TextAlign.Center)
                    Spacer(Modifier.height(12.dp))
                    OutlinedButton(onClick = {
                        cameraError = null
                        cameraAttempt++
                    }) { Text("Retry camera") }
                }
            }
        }
        Text(
            "Frame the vehicle and plate, then capture.",
            modifier = Modifier.fillMaxWidth().padding(top = 18.dp, bottom = 12.dp),
            color = Paper.copy(alpha = 0.75f),
            textAlign = TextAlign.Center,
        )
        recordError?.let {
            Text(
                "Photo saved, but its record status could not be saved: $it",
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                color = Color(0xFFFFB4AB),
                textAlign = TextAlign.Center,
            )
        }
        Button(
            onClick = {
                val useCase = imageCapture ?: return@Button
                if (isCapturing) return@Button
                isCapturing = true
                val directory = File(context.filesDir, "captures")
                if (!directory.exists() && !directory.mkdirs()) {
                    isCapturing = false
                    cameraError = "Could not prepare photo storage. Try again."
                    return@Button
                }
                val timestamp = System.currentTimeMillis()
                var suffix = 0
                var photo = File(directory, "vehicle_$timestamp.jpg")
                while (photo.exists()) {
                    suffix++
                    photo = File(directory, "vehicle_${timestamp}_$suffix.jpg")
                }
                if (!runCatching { photo.createNewFile() }.getOrDefault(false)) {
                    isCapturing = false
                    cameraError = "Could not create a new photo file. Try again."
                    return@Button
                }
                var photoSaved = false
                try {
                    useCase.targetRotation = view.display?.rotation ?: Surface.ROTATION_0
                    val options = ImageCapture.OutputFileOptions.Builder(photo).build()
                    useCase.takePicture(
                        options,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                photoSaved = true
                                isCapturing = false
                                onCaptured(photo.absolutePath)
                            }

                            override fun onError(exception: ImageCaptureException) {
                                isCapturing = false
                                if (!photoSaved) removeIncompleteCapture(photo)
                                cameraError = exception.message ?: "Photo capture failed. Try again."
                            }
                        },
                    )
                } catch (error: Exception) {
                    isCapturing = false
                    if (!photoSaved) removeIncompleteCapture(photo)
                    cameraError = error.message ?: "Photo capture failed. Try again."
                }
            },
            enabled = imageCapture != null && !isCapturing && cameraError == null,
            modifier = Modifier.align(Alignment.CenterHorizontally).padding(bottom = 24.dp).height(62.dp).width(210.dp),
            shape = RoundedCornerShape(32.dp),
        ) {
            if (isCapturing) {
                CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(10.dp))
                Text("Capturing...")
            } else {
                Text("Capture photo", fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun RecordsScreen(
    pending: List<PlateRecord>,
    history: List<PlateRecord>,
    selectedPage: RecordPage,
    onPageSelected: (RecordPage) -> Unit,
    onCamera: () -> Unit,
    onSelect: (PlateRecord) -> Unit,
    onRetry: (PlateRecord) -> Unit,
) {
    Column(Modifier.fillMaxSize().background(Ink)) {
        ScreenHeader(selectedPage.name.lowercase().replaceFirstChar { it.uppercase() }, onCamera, actionLabel = "Camera")
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            listOf(RecordPage.PENDING, RecordPage.HISTORY).forEach { page ->
                TextButton(onClick = { onPageSelected(page) }, modifier = Modifier.weight(1f)) {
                    Text(
                        page.name.lowercase().replaceFirstChar { it.uppercase() },
                        color = if (page == selectedPage) Accent else Paper.copy(alpha = 0.7f),
                        fontWeight = if (page == selectedPage) FontWeight.Bold else FontWeight.Normal,
                    )
                }
            }
        }
        val records = if (selectedPage == RecordPage.PENDING) pending else history
        if (records.isEmpty()) {
            Column(
                Modifier.weight(1f).fillMaxWidth().padding(28.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    if (selectedPage == RecordPage.PENDING) "No pending records" else "No history records",
                    color = Paper,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    if (selectedPage == RecordPage.PENDING) "Captured vehicle photos awaiting review appear here."
                    else "Submitted recognition records appear here.",
                    color = Paper.copy(alpha = 0.72f),
                    textAlign = TextAlign.Center,
                )
            }
        } else if (selectedPage == RecordPage.PENDING) {
            Column(Modifier.weight(1f).fillMaxWidth()) {
                Text(
                    "Select a saved vehicle photo to mark its plate region.",
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    color = Paper.copy(alpha = 0.75f),
                )
                LazyVerticalGrid(
                    columns = GridCells.Adaptive(minSize = 160.dp),
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    items(items = pending, key = { it.id }) { record ->
                        Column(
                            Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Color(0xFF1A2730)).padding(10.dp),
                        ) {
                            Thumbnail(
                                path = record.originalPath,
                                label = "Original vehicle photo",
                                modifier = Modifier.fillMaxWidth().height(132.dp),
                            )
                            Text(
                                "Status: PENDING",
                                modifier = Modifier.padding(top = 6.dp),
                                color = Paper.copy(alpha = 0.8f),
                            )
                            Button(
                                onClick = { onSelect(record) },
                                modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                            ) {
                                Text("Review plate")
                            }
                        }
                    }
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                columnItems(items = history, key = { it.id }) { record ->
                    HistoryRecordCard(record, onRetry)
                }
            }
        }
    }
}

@Composable
private fun HistoryRecordCard(record: PlateRecord, onRetry: (PlateRecord) -> Unit) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Color(0xFF1A2730)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Thumbnail(
            path = record.originalPath,
            label = "Original vehicle photo",
            modifier = Modifier.fillMaxWidth().height(180.dp),
        )
        Text("Status: ${record.state.name}", color = Accent, fontWeight = FontWeight.SemiBold)
        Text("Offence: ${record.offence ?: "Unavailable"}", color = Paper)
        Text("Plate number: ${record.plateNumber ?: "Unavailable"}", color = Paper)
        Text("Confidence: ${record.confidence?.toString() ?: "Unavailable"}", color = Paper)
        Text("API status: ${record.apiStatus ?: "Unavailable"}", color = Paper)
        record.validIndianFormat?.let {
            Text(
                when {
                    record.plateNumber.isNullOrBlank() -> "Indian-format flag: $it (no plate text to validate)"
                    it -> "Indian-format check: passed"
                    else -> "Indian-format check: did not pass"
                },
                color = if (record.state == RecordState.INVALID_FORMAT) Color(0xFFFFD08A) else Paper,
            )
        }
        if (record.state == RecordState.INVALID_FORMAT) {
            Text(
                "OCR text is preserved above for review; it does not pass the existing Indian-format check.",
                color = Color(0xFFFFD08A),
            )
        }
        record.errorDetails?.let { Text("Details: $it", color = Color(0xFFFFB4AB)) }
        record.cropPath?.takeIf { File(it).isFile }?.let {
            Thumbnail(
                path = it,
                label = "Selected plate crop",
                modifier = Modifier.fillMaxWidth().height(110.dp),
            )
        }
        if (record.state == RecordState.FAILED || record.state == RecordState.INVALID_FORMAT) {
            OutlinedButton(onClick = { onRetry(record) }, modifier = Modifier.fillMaxWidth()) {
                Text("Retry recognition")
            }
        }
    }
}

@Composable
private fun CameraPreview(
    modifier: Modifier,
    onCaptureReady: (ImageCapture?) -> Unit,
    onError: (String?) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    val currentCaptureReady by rememberUpdatedState(onCaptureReady)
    val currentError by rememberUpdatedState(onError)
    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()
    }

    AndroidView(factory = { previewView }, modifier = modifier)
    DisposableEffect(lifecycleOwner, previewView, imageCapture) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        var disposed = false
        val listener = Runnable {
            if (!disposed) {
                try {
                    val cameraProvider = providerFuture.get()
                    provider = cameraProvider
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture,
                    )
                    currentCaptureReady(imageCapture)
                    currentError(null)
                } catch (error: Exception) {
                    currentCaptureReady(null)
                    currentError(error.message ?: "Could not start the camera preview.")
                }
            }
        }
        providerFuture.addListener(listener, ContextCompat.getMainExecutor(context))
        onDispose {
            disposed = true
            provider?.unbindAll()
            currentCaptureReady(null)
        }
    }
}

@Composable
private fun ReviewScreen(
    photoPath: String,
    selection: NormalizedRect?,
    onSelectionChanged: (NormalizedRect?) -> Unit,
    onClear: () -> Unit,
    error: String?,
    submitting: Boolean,
    onConfirm: (String) -> Unit,
    onCancel: () -> Unit,
) {
    var offence by rememberSaveable(photoPath) { mutableStateOf("") }
    Column(Modifier.fillMaxSize().background(Ink).padding(horizontal = 16.dp)) {
        ScreenHeader("Review photo", onCancel, actionLabel = "Cancel")
        Text(
            "Tap two corners or drag to mark the plate. Drag inside the box to move it.",
            modifier = Modifier.fillMaxWidth().padding(bottom = 10.dp),
            color = Paper.copy(alpha = 0.78f),
            textAlign = TextAlign.Center,
            fontSize = 14.sp,
        )
        SelectablePhoto(
            path = photoPath,
            selection = selection,
            onSelectionChanged = onSelectionChanged,
            modifier = Modifier.weight(1f).fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Color.Black),
        )
        OutlinedTextField(
            value = offence,
            onValueChange = { offence = it },
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
            label = { Text("Alleged offence description", color = Paper.copy(alpha = 0.75f)) },
            minLines = 2,
            maxLines = 3,
            enabled = !submitting,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Paper,
                unfocusedTextColor = Paper,
                cursorColor = Accent,
                focusedBorderColor = Accent,
                unfocusedBorderColor = Paper.copy(alpha = 0.7f),
                focusedLabelColor = Accent,
                unfocusedLabelColor = Paper.copy(alpha = 0.75f),
            ),
        )
        error?.let {
            Text(it, modifier = Modifier.padding(top = 6.dp), color = Color(0xFFFFB4AB))
        }
        Row(
            Modifier.fillMaxWidth().padding(vertical = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedButton(onClick = onClear, modifier = Modifier.weight(1f), enabled = selection != null && !submitting) {
                Text("Clear selection")
            }
            Button(
                onClick = { onConfirm(offence.trim()) },
                modifier = Modifier.weight(1f),
                enabled = selection.isUsable() && offence.isNotBlank() && !submitting,
            ) {
                if (submitting) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                }
                Text(if (submitting) "Submitting..." else "Confirm & submit")
            }
        }
    }
}

@Composable
private fun SelectablePhoto(
    path: String,
    selection: NormalizedRect?,
    onSelectionChanged: (NormalizedRect?) -> Unit,
    modifier: Modifier,
) {
    BoxWithConstraints(modifier) {
        val density = LocalDensity.current
        val targetWidth = with(density) { maxWidth.roundToPx() }.coerceAtLeast(1)
        val targetHeight = with(density) { maxHeight.roundToPx() }.coerceAtLeast(1)
        val loadState by produceState<ImageLoadState>(
            initialValue = ImageLoadState.Loading,
            key1 = path,
            key2 = targetWidth,
            key3 = targetHeight,
        ) {
            value = withContext(Dispatchers.IO) {
                runCatching { ImageFiles.decodePreview(path, targetWidth, targetHeight) }
                    .fold({ ImageLoadState.Loaded(it) }, { ImageLoadState.Failed(it.message ?: "Photo could not be loaded") })
            }
        }

        when (val state = loadState) {
            ImageLoadState.Loading -> CircularProgressIndicator(Modifier.align(Alignment.Center), color = Accent)
            is ImageLoadState.Failed -> Text(state.message, Modifier.align(Alignment.Center).padding(20.dp), color = Paper)
            is ImageLoadState.Loaded -> {
                val bitmap = state.bitmap
                val containerWidth = constraints.maxWidth.toFloat()
                val containerHeight = constraints.maxHeight.toFloat()
                val scale = min(containerWidth / bitmap.width, containerHeight / bitmap.height)
                val drawWidth = bitmap.width * scale
                val drawHeight = bitmap.height * scale
                val imageBounds = Rect(
                    (containerWidth - drawWidth) / 2f,
                    (containerHeight - drawHeight) / 2f,
                    (containerWidth + drawWidth) / 2f,
                    (containerHeight + drawHeight) / 2f,
                )
                ImageSelectionOverlay(
                    bitmap = bitmap,
                    imageBounds = imageBounds,
                    selection = selection,
                    onSelectionChanged = onSelectionChanged,
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
    }
}

@Composable
private fun ImageSelectionOverlay(
    bitmap: android.graphics.Bitmap,
    imageBounds: Rect,
    selection: NormalizedRect?,
    onSelectionChanged: (NormalizedRect?) -> Unit,
    modifier: Modifier,
) {
    var tapAnchor by remember(bitmap.width, bitmap.height) { mutableStateOf<Offset?>(null) }
    val currentSelection by rememberUpdatedState(selection)
    val currentTapAnchor by rememberUpdatedState(tapAnchor)
    val currentOnSelectionChanged by rememberUpdatedState(onSelectionChanged)
    LaunchedEffect(selection) {
        if (selection == null) tapAnchor = null
    }
    Box(
        modifier.pointerInput(bitmap.width, bitmap.height, imageBounds) {
            val touchSlop = viewConfiguration.touchSlop
            fun pointAt(position: Offset, clamp: Boolean): Offset? {
                if (!clamp && !imageBounds.contains(position)) return null
                val x = position.x.coerceIn(imageBounds.left, imageBounds.right)
                val y = position.y.coerceIn(imageBounds.top, imageBounds.bottom)
                return Offset(
                    ((x - imageBounds.left) / imageBounds.width).coerceIn(0f, 1f),
                    ((y - imageBounds.top) / imageBounds.height).coerceIn(0f, 1f),
                )
            }
            fun makeRect(first: Offset, second: Offset) = NormalizedRect(
                min(first.x, second.x),
                min(first.y, second.y),
                max(first.x, second.x),
                max(first.y, second.y),
            )
            fun selectionBounds(value: NormalizedRect) = Rect(
                imageBounds.left + value.left * imageBounds.width,
                imageBounds.top + value.top * imageBounds.height,
                imageBounds.left + value.right * imageBounds.width,
                imageBounds.top + value.bottom * imageBounds.height,
            )

            awaitEachGesture {
                val down = awaitFirstDown(requireUnconsumed = false)
                val start = down.position
                val startPoint = pointAt(start, clamp = false) ?: return@awaitEachGesture
                val initialSelection = currentSelection
                val selectionToMove = initialSelection?.takeIf { selectionBounds(it).contains(start) }
                var movingSelection = selectionToMove
                var dragging = false
                var previous = start
                var latest = start
                while (true) {
                    val event = awaitPointerEvent()
                    val change = event.changes.firstOrNull { it.id == down.id } ?: break
                    latest = change.position
                    if (!dragging && (latest - start).getDistance() > touchSlop) {
                        dragging = true
                        tapAnchor = null
                        if (selectionToMove == null) currentOnSelectionChanged(makeRect(startPoint, startPoint))
                    }
                    if (dragging) {
                        val current = pointAt(latest, clamp = true) ?: startPoint
                        if (selectionToMove != null) {
                            val dx = (latest.x - previous.x) / imageBounds.width
                            val dy = (latest.y - previous.y) / imageBounds.height
                            val width = selectionToMove.right - selectionToMove.left
                            val height = selectionToMove.bottom - selectionToMove.top
                            val latestSelection = movingSelection ?: selectionToMove
                            val left = latestSelection.left + dx
                            val top = latestSelection.top + dy
                            val clampedLeft = left.coerceIn(0f, 1f - width)
                            val clampedTop = top.coerceIn(0f, 1f - height)
                            val updatedSelection = NormalizedRect(
                                clampedLeft,
                                clampedTop,
                                clampedLeft + width,
                                clampedTop + height,
                            )
                            movingSelection = updatedSelection
                            currentOnSelectionChanged(updatedSelection)
                        } else {
                            currentOnSelectionChanged(makeRect(startPoint, current))
                        }
                    }
                    previous = latest
                    if (!change.pressed) {
                        change.consume()
                        break
                    }
                    change.consume()
                }
                if (!dragging) {
                    pointAt(latest, clamp = false)?.let { point ->
                        val anchor = currentTapAnchor
                        if (anchor == null) {
                            tapAnchor = point
                            currentOnSelectionChanged(NormalizedRect(point.x, point.y, point.x, point.y))
                        } else {
                            tapAnchor = null
                            currentOnSelectionChanged(makeRect(anchor, point))
                        }
                    }
                }
            }
        },
    ) {
        androidx.compose.foundation.Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "Captured vehicle photo",
            modifier = Modifier.fillMaxSize(),
            contentScale = androidx.compose.ui.layout.ContentScale.Fit,
        )
        Canvas(Modifier.fillMaxSize()) {
            selection?.let { selected ->
                val left = imageBounds.left + selected.left * imageBounds.width
                val top = imageBounds.top + selected.top * imageBounds.height
                val right = imageBounds.left + selected.right * imageBounds.width
                val bottom = imageBounds.top + selected.bottom * imageBounds.height
                if (right > left && bottom > top) {
                    val rect = Rect(left, top, right, bottom)
                    val shade = Path().apply {
                        fillType = PathFillType.EvenOdd
                        addRect(imageBounds)
                        addRect(rect)
                    }
                    drawPath(shade, Color.Black.copy(alpha = 0.45f))
                    drawRect(Accent, Offset(left, top), androidx.compose.ui.geometry.Size(right - left, bottom - top), style = Stroke(3.dp.toPx()))
                    val handleRadius = 5.dp.toPx()
                    listOf(Offset(left, top), Offset(right, top), Offset(left, bottom), Offset(right, bottom)).forEach {
                        drawCircle(Color.White, handleRadius + 1.dp.toPx(), it)
                        drawCircle(Accent, handleRadius, it)
                    }
                } else {
                    drawCircle(Accent, 8.dp.toPx(), Offset(left, top))
                }
            }
        }
    }
}

@Composable
private fun Thumbnail(path: String, label: String, modifier: Modifier) {
    BoxWithConstraints(modifier.clip(RoundedCornerShape(16.dp)).background(Color(0xFF1A2730)).padding(8.dp)) {
        Column(Modifier.fillMaxSize()) {
            Text(label, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp), color = Paper, fontWeight = FontWeight.Medium)
            val density = LocalDensity.current
            val targetWidth = with(density) { this@BoxWithConstraints.maxWidth.roundToPx() }.coerceAtLeast(1)
            val targetHeight = with(density) { this@BoxWithConstraints.maxHeight.roundToPx() }.coerceAtLeast(1)
            val loadState by produceState<ImageLoadState>(ImageLoadState.Loading, path, targetWidth, targetHeight) {
                value = withContext(Dispatchers.IO) {
                    runCatching { ImageFiles.decodePreview(path, targetWidth, targetHeight) }
                        .fold({ ImageLoadState.Loaded(it) }, { ImageLoadState.Failed(it.message ?: "Photo unavailable") })
                }
            }
            when (val state = loadState) {
                ImageLoadState.Loading -> CircularProgressIndicator(Modifier.align(Alignment.CenterHorizontally).padding(8.dp), color = Accent)
                is ImageLoadState.Failed -> Text(state.message, Modifier.padding(8.dp), color = Paper)
                is ImageLoadState.Loaded -> androidx.compose.foundation.Image(
                    bitmap = state.bitmap.asImageBitmap(),
                    contentDescription = label,
                    modifier = Modifier.fillMaxWidth().weight(1f),
                    contentScale = androidx.compose.ui.layout.ContentScale.Fit,
                )
            }
        }
    }
}

@Composable
private fun ScreenHeader(title: String, onCamera: () -> Unit, actionLabel: String) {
    Row(
        Modifier.fillMaxWidth().height(58.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(title, color = Paper, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
        TextButton(onClick = onCamera) { Text(actionLabel) }
    }
}

private fun NormalizedRect?.isUsable(): Boolean =
    this != null && right - left >= 0.01f && bottom - top >= 0.01f

private fun openApplicationSettings(context: android.content.Context) {
    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
        data = Uri.fromParts("package", context.packageName, null)
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
    context.startActivity(intent)
}

private fun removeIncompleteCapture(photo: File) {
    runCatching {
        if (photo.isFile) photo.delete()
    }
}

private sealed interface ImageLoadState {
    data object Loading : ImageLoadState
    data class Loaded(val bitmap: android.graphics.Bitmap) : ImageLoadState
    data class Failed(val message: String) : ImageLoadState
}
