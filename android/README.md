# Plate Capture Android Prototype

This is a separate Kotlin and Jetpack Compose Android app. It sends a confirmed plate crop to the existing Flask ALPR API; the Python recognition engine is unchanged.

## What it does

- Requests only camera permission and opens a CameraX back-camera preview.
- Captures a quality-oriented JPEG into the app's private `files/captures` directory.
- Shows an orientation-correct review image without editing the original photo.
- Lets the user mark a region with a drag or two taps, move the selection by dragging inside it, redraw it by starting a drag outside it, or clear it.
- Saves the selected region as a separate JPEG, scaled to at most 1600 pixels on its longest edge.
- After plate selection and offence entry, saves the crop separately and enqueues persistent WorkManager recognition using multipart field `image` at `http://192.168.1.71:5000/recognize`.
- Returns to Pending after submission. History shows PROCESSING, COMPLETED, INVALID_FORMAT, and FAILED records with API status and available results or error details. Failed and invalid-format attempts can be retried manually.

The selected box is kept as normalized image coordinates across rotation and screen resizing. The original photo and region crop are separate private files. The development endpoint uses cleartext HTTP and is configured in `RecognitionApi.kt`.

## Build configuration

The project uses Android Gradle Plugin 9.4.1, Gradle 9.6.0, the AGP built-in Kotlin support, Compose Compiler plugin 2.4.10, Compose BOM 2026.09.00, and CameraX 1.6.2. It compiles against Android API 37, targets API 36, and supports API 23 and later. It requests `CAMERA` and `INTERNET`; cleartext traffic is enabled for the development API. AndroidX adds its internal signature-level receiver permission; photos remain in app-private storage. Java and Kotlin bytecode target 17.

Open the `android` directory in Android Studio with Gradle JDK 17 or newer and install Android SDK Platform 37 and Build Tools 36.0.0 through SDK Manager. Then sync and run the `app` configuration. The included Gradle wrapper can build from a terminal:

```powershell
cd android
.\gradlew.bat :app:assembleDebug
```

## Install on a physical phone

1. Install Android Studio and a supported Gradle JDK (17 or newer), then install Android SDK Platform 37 and Build Tools 36.0.0 in SDK Manager.
2. Enable Developer options and USB debugging on the phone, connect it by USB, and accept its debugging authorization prompt.
3. From `android/`, run `adb devices` and confirm the phone appears as `device`.
4. From `android/`, run `.\gradlew.bat :app:installDebug`, or select the phone in Android Studio and run the `app` configuration.
5. Ensure the Flask server is reachable at `192.168.1.71:5000` from the phone, launch **Plate Capture**, allow camera access, capture a vehicle, select its plate region, enter an offence description, and confirm. The app returns to Pending while recognition runs; open History to view the status and result, or retry a FAILED or INVALID_FORMAT attempt.

For a different development host, update `ENDPOINT` in `RecognitionApi.kt`. Keep cleartext traffic enabled only for development; production deployment should use HTTPS.
