# Indian License Plate Recognition (DSP / OpenCV)

This project takes a vehicle photograph, automatically proposes plate locations, rectifies and enhances the best plate crop, and reads the registration string. Plate location and image cleanup use classical image processing. OCR uses RapidOCR's compact ONNX text recognizer, with optional Tesseract and a built-in OpenCV glyph-template fallback.

## Run

```powershell
python -m pip install -r requirements.txt
python -m alpr car01.webp --output alpr_output
```

RapidOCR and its OCR model files are installed by the requirements above. For an alternate OCR path, install the Tesseract executable and the optional `pytesseract` binding; the program will use it when RapidOCR is unavailable. If neither OCR engine is installed, generated OpenCV Hershey-font templates provide a best-effort fallback for clean, front-facing plates.

Output files include the annotated vehicle image, plate crop, contrast-normalized threshold image, edge/gradient/threshold maps, ranked proposal image, and `results.json`. Boxes in JSON are `(x, y, width, height)` in the source image.

## App-facing Python interface

Pass a decoded OpenCV BGR image array to `alpr.recognize_for_app(image)`. It returns an `AppResult` with the plate number, confidence, Indian-format flag, source-image bounding box `(x, y, width, height)`, crop, OCR method, and processing status (`success`, `plate_not_found`, `recognition_failed`, or `error`). The default app call does not retain or write diagnostic images. Set `include_diagnostics=True` to include localization diagnostics in the result. The input evidence array is not modified; CLI output continues to save a separate `01_input.jpg` and annotated `06_detected.jpg`. At the lower-level API, `recognize_image(..., include_diagnostics=False, output_dir=...)` saves the original under `evidence/original.jpg` and omits diagnostic image files.

## Android prototype

The separate Kotlin and Jetpack Compose app is in [`android/`](android/README.md). Captured originals appear in Pending. After plate-region selection and offence entry, the app saves a separate crop, queues recognition with persistent WorkManager work, and returns to Pending. Results and processing status appear in History; failed attempts can be retried manually. The app sends the selected crop to the Flask `/recognize` API.

## Processing design

1. **Normalize and enhance:** convert to grayscale, apply CLAHE, and bilateral filtering to reduce noise without erasing glyph edges.
2. **Find plate candidates:** combine horizontal Sobel gradients closed with a scale-relative rectangular kernel, Canny edges with morphological closing, adaptive thresholding, groups of aligned character-sized components, and full-scene OCR text boxes that resemble registration strings. Geometry accepts both broad one-line ratios and near-square or tall two-line layouts.
3. **Rank candidates:** score plate-like aspect ratio, reflective bright area, dark-on-light contrast, edge density, contour fill, and the count of character-sized connected components. Overlapping proposals are suppressed.
4. **Crop and rectify:** add a small image-relative margin and search within the candidate for a large four-sided contour. Apply a perspective warp when a plausible quadrilateral is present; otherwise retain the crop.
5. **Prepare OCR variants:** CLAHE, edge-preserving denoising, unsharp masking, cubic upscaling, Otsu thresholding, and adaptive thresholding provide several views of the glyphs.
6. **Read and rank:** RapidOCR reads multiple enhanced variants and stitches text regions in reading order. Scene-level and crop-level OCR are aligned to recover characters missed at small scale. Optional Tesseract tests PSM 7 and 6 with an alphanumeric whitelist. If both are unavailable, the local fallback segments connected components, orders rows for two-line plates, and classifies glyphs against OpenCV-rendered templates. Indian registration syntax helps rank candidate readings; other plate-like letter/digit formats can still be returned.
7. **Save evidence:** export processing maps, candidate boxes, selected crop, processed plate, annotated input, and per-candidate scores.

## Example run

On `car01.webp`, the current run detects box `(221, 179, 50, 18)` and reads **RJ25CA6667**. On the distant, rainy `car03.jpeg` photo, it detects box `(779, 520, 94, 30)` and reads **KL07CX4823**. On the dark `image.png`, it detects box `(488, 942, 54, 21)` and reads **SJK9220**; that string does not match the Indian-format check. Each output folder contains its proposal map, plate crops, threshold/edge maps, annotated image, and machine-readable result summary.

## Limitations

Classical localization can be confused by grille slats, signs, or bright trim with plate-like shape and contrast. Small or oblique plates, severe glare, motion blur, dirt, decorative fonts, nonstandard spacing, and occlusion can defeat segmentation or OCR. The template fallback is less reliable than a trained OCR engine and expects recognizable uppercase Latin letters/digits. The format check is a useful ranking clue, not proof that the string is a valid issued registration. The system is an educational prototype and is not calibrated for enforcement or identity-sensitive use.
