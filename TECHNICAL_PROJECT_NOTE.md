# Technical Project Note: Vehicle License Plate Recognition

This note describes the implementation currently in `alpr/`, including the later full-image OCR proposal path. It records the behavior of the code as run on the current `car01.webp`; it is not a design proposal or a claim that every plate layout has been validated.

## 1. Project Objective

The program accepts a vehicle photograph and attempts to:

- locate one or more plate-like regions automatically;
- crop and optionally perspective-correct a selected region;
- recognize its alphanumeric registration string;
- save the detected box, intermediate images, annotated photo, and a JSON report.

The normal input is a file path passed to the CLI. The Python API also accepts an OpenCV image array. File inputs are decoded as BGR with `cv2.imread`; grayscale arrays are converted to BGR, while color arrays are treated as BGR.

The main return value is an `ALPRResult` with `plate_number`, `valid_indian_format`, `confidence`, `ocr_method`, `plate_box`, crops, diagnostics, candidates, and an annotated image. The CLI prints a JSON subset of those fields and writes image and JSON artifacts when an output directory is supplied.

## 2. Complete System Pipeline

```text
Image path / BGR array
  → decode, reject empty input, resize so longest side is at most 1600 px
  → classical localization maps: CLAHE + bilateral filter
  → Sobel/morphology, Canny/morphology, adaptive threshold contours
  → adaptive-threshold connected components grouped into aligned glyph rows
  → RapidOCR over the full image; retain text boxes with registration-like strings
  → combine and sort classical and OCR-derived candidate boxes
  → crop each of up to 15 candidates and try perspective rectification
  → grayscale/CLAHE/bilateral/unsharp processing, resize, Otsu and adaptive variants
  → RapidOCR crop reading; optionally Tesseract and/or OpenCV template fallback
  → merge full-scene and crop OCR strings when a scene OCR hint exists
  → rank evaluated candidates, stop early on a strong registration-like read
  → return text/box/confidence and save intermediate images + results.json
```

The full-image OCR pass is a general text detector/recognizer used as an additional proposal source. The project does not use a dedicated pretrained license-plate detector or end-to-end ALPR model. The classical candidate paths remain active and are combined with those text proposals.

## 3. Image Preprocessing

### Input sizing

`recognize_image` downsizes images whose longest side exceeds 1600 pixels, using `INTER_AREA`. It does not enlarge smaller input images at this stage. `localize_plates` contains the same maximum-dimension safeguard for direct callers. Candidate boxes are converted back to source-image coordinates by the pipeline's resize scale.

### Localization preprocessing

| Operation | Actual settings | Use in this program |
|---|---|---|
| BGR to grayscale | `cv2.COLOR_BGR2GRAY` | Supplies a single intensity channel for edge, threshold, and contrast processing. |
| CLAHE | `clipLimit=2.2`, `tileGridSize=(8,8)` | Local contrast enhancement before proposal generation. |
| Bilateral filter | `d=7`, `sigmaColor=55`, `sigmaSpace=55` | Smooths the enhanced image while retaining edges. |
| Horizontal Sobel | `CV_32F`, `dx=1`, `dy=0`, `ksize=3`, then `convertScaleAbs` | Emphasizes vertical intensity transitions around character strokes and plate boundaries. |
| Otsu threshold of Sobel magnitude | `THRESH_BINARY`, automatic Otsu threshold | Creates a binary horizontal-gradient proposal map. |
| Horizontal close/open | Rectangular kernel `(max(9, int(W*0.045)|1), max(3, int(H*0.012)|1))`; close, 2 iterations; then open with `3x3` rectangle | Joins nearby horizontal responses and removes small responses. The `|1` expressions make kernel dimensions odd. |
| Canny edges | thresholds `55`, `165` | Creates an additional edge map. |
| Canny-map morphology | Close with the same scale-relative horizontal kernel, 2 iterations; dilate with `3x3` rectangle, 1 iteration | Joins nearby plate-like edges. |
| Adaptive threshold | Gaussian method, binary inverse, block size `21`, C=`7` | Finds local dark-on-light structure; a copy is horizontally closed once for contour proposals. |

The outputs of these operations feed contour candidate generation; they are also saved as diagnostics.

### Glyph-component proposal preprocessing

The separate aligned-glyph candidate path applies CLAHE with `clipLimit=2.0`, grid `(8,8)`, then Gaussian adaptive inverse thresholding with C=`5`. Its odd block size is `max(15, int(0.065 * min(H,W)) | 1)`, clamped to an odd value no larger than the image's shorter side.

Connected components are retained when their width is between `max(2, 0.0025*W)` and `0.075*W`; height is between `max(3, 0.008*min(H,W))` and `0.13*H`; area is at least 5 pixels; and component width/height is in `[0.12, 1.4]`. Seed components are grouped by similar vertical center and height, then split into horizontal runs using a gap limit of 1.35 times character height. This stage proposes text-row regions; it does not recognize the characters.

## 4. License Plate Localization

### Classical candidate generation

`localize_plates` finds contours using `RETR_LIST` and `CHAIN_APPROX_SIMPLE` on three maps:

1. the closed/opened horizontal Sobel map;
2. the closed and dilated Canny map;
3. the horizontally closed adaptive-threshold map.

Each contour's axis-aligned `boundingRect` becomes a candidate. The contour area is retained as a rectangularity input to scoring.

The glyph path also proposes a box around runs containing 5–13 aligned components. Each row gets horizontal padding `max(2, 0.13*row_width)` and vertical padding `max(2, 0.40*median_character_height, 0.006*min(H,W))`. Pairs of rows may form a two-line proposal when their center spacing is between 0.65 and 3 character heights and their horizontal overlap is at least 28% of the narrower row width; paired boxes receive 3 px horizontal and 2 px vertical padding.

If all contour/glyph candidate paths yield no selected candidate, there is a final bright-region fallback: Otsu binary threshold on the CLAHE grayscale image, horizontal closing twice, and external-contour boxes.

### Candidate geometry and visual checks

The `_score_box` filter accepts aspect ratios from `0.68` to `7.5`; width at least `max(18, 0.035*image_width)`; height at least `max(9, 0.012*min(H,W))`; height no greater than `0.46*H`; and area no greater than 30% of the image area. The aspect prior treats ratios at least `2.1` as single-line-like, centered on 4.0, and lower ratios as two-line-like, centered on 1.28.

For each surviving ROI, the score components are:

- **Aspect score:** exponential distance from the single- or two-line aspect target.
- **Bright-background score:** fraction of grayscale pixels above 150, with a target fraction of 0.60.
- **Contrast score:** standard deviation divided by 58, capped at 1.
- **Character score:** Gaussian blur `3x3`, Otsu inverse threshold, then count connected components with height 18–88% of the ROI height, width 1.2–24% of the ROI width, and area at least 5. The score favors a count of 8; if the ROI fraction below 105 is less than 0.015, this score is multiplied by 0.25.
- **Rectangularity score:** `0.5 + 0.5 * min(contour_area / box_area, 1)`.
- **Edge score:** mean Canny response at thresholds 60/150 divided by 42, capped at 1.

The weighted sum is `0.27*aspect + 0.27*brightness + 0.16*contrast + 0.18*characters + 0.07*rectangularity + 0.05*edges`.

After generation, classical candidates are sorted by this score. Greedy non-maximum suppression keeps a candidate only when its IoU with every retained box is below `0.68`; at most 30 are returned by the localizer.

### Full-image OCR proposals and final candidate selection

The pipeline separately runs RapidOCR on the full working image. Its detected text boxes are retained when normalized text matches either the Indian pattern or the generic pattern `[A-Z]{2,3}[0-9]{3,4}`, and the box has aspect ratio 1.45–9, width at least `max(12, 0.025*image_width)`, height at least `max(6, 0.006*min(H,W))`, and area no more than 12% of the image.

Each retained OCR proposal receives `0.55*OCR_confidence + 0.25*shape_score + 0.20`, where `shape_score = exp(-abs(log(aspect/4))/1.5)`. These proposals are appended to the classical list and all candidates are sorted by score. The pipeline does not run another IoU suppression pass after appending OCR proposals.

The CLI default is to OCR at most 15 candidates (`--max-candidates` overrides this). For each, the crop is read and a combined ranking score is formed from `0.42*candidate_score + 0.58*OCR_confidence`, plus `0.15` when the text looks registration-like, capped at 1. The pipeline stops evaluating lower-ranked candidates when a registration-like read has OCR confidence at least `0.72`. Among evaluated candidates it sorts registration-like reads before other text, then by combined score. The final reported `confidence` omits the 0.15 ranking bonus and is not calibrated as a probability.

## 5. Plate Rectification

The selected candidate's OCR crop uses source-specific margins: no extra margin for aligned-glyph proposals; 16% of box width and 35% of box height for OCR text boxes, with minimum padding of 2 px horizontally and 3 px vertically; otherwise 3.5% width and 10% height, with a 1 px minimum on each axis.

`rectify_plate` then:

1. converts the crop to grayscale and applies CLAHE (`clipLimit=2.0`, grid `(4,4)`);
2. Gaussian-blurs with `3x3`, then runs Canny at `35/120`;
3. closes edges with `3x3` ones for 2 iterations;
4. searches `RETR_LIST` contours for a 4-vertex approximation using epsilon `0.025 * perimeter`;
5. keeps quadrilaterals with area at least 38% of the crop and a min-area-rectangle aspect ratio from 0.65 to 7.8;
6. ranks them by `coverage * (0.65 + 0.35*fill)`, where fill is polygon area divided by its bounding-rectangle area;
7. warps the highest-ranked `minAreaRect` box to a rectangular output using `getPerspectiveTransform` and `warpPerspective` with cubic interpolation and replicated borders.

If no quadrilateral passes those checks, the original candidate crop is used unchanged. A portrait result is rotated to landscape only when the original candidate crop's aspect ratio exceeds 2.1; near-square/tall two-line candidates are therefore not forcibly rotated.

## 6. Plate-Specific Processing

After rectification, `preprocess_plate` applies these operations in order:

1. Convert to grayscale if needed.
2. Apply CLAHE (`clipLimit=2.5`, grid `(4,4)`).
3. Apply bilateral filtering (`d=5`, `sigmaColor=28`, `sigmaSpace=28`).
4. Sharpen using `1.7*smooth - 0.7*GaussianBlur(smooth, sigma=1.1)`, plus OpenCV's `addWeighted` default gamma of 0.
5. Resize with cubic interpolation. Target height is 112 px when the input crop height is below 32 px, otherwise 96 px. Scale is clamped to `[2,8]` and width is scaled proportionally.
6. Add a replicated border of 8 pixels on top/bottom and 10 pixels on left/right.
7. Create a normal Otsu binary image and a Gaussian adaptive binary image (block size 31, C=9). Their inverse images are also returned for template segmentation.

These are variants, not a sequence of threshold operations: RapidOCR and Tesseract receive grayscale/Otsu/adaptive versions; the local template reader uses the inverse binary versions. The saved `05_plate_processed.png` is the Otsu variant returned by `recognize_crop`, falling back to enlarged grayscale only if Otsu is unavailable.

## 7. Character Recognition

### Primary OCR: RapidOCR

The required `rapidocr-onnxruntime` package provides a general scene-text detection/recognition model. The code lazily constructs one `RapidOCR()` engine and reuses it.

- **Scene pass:** `detect_scene_text` sends the whole working image to RapidOCR. It filters returned text polygons by registration-like text and geometric limits; qualifying boxes become additional candidates.
- **Crop pass:** `_rapidocr_read` sends the `gray`, `otsu`, and `adaptive` plate variants separately. It sorts text boxes by mean y then mean x and concatenates their strings. It averages the boxes' OCR confidence scores. The Indian format receives a 0.18 bonus only while choosing among these variants; the returned OCR confidence is the raw mean.
- If crop RapidOCR confidence is at least 0.75 and the text matches the Indian regex, `recognize_crop` returns it immediately. If confidence is at least 0.50 and no `tesseract` executable is discoverable, it returns the RapidOCR result even if it is not Indian-format text.

### Scene/crop text reconciliation

When an OCR-derived proposal carries a scene-level text hint, `merge_text_reads` compares the normalized hint and crop read with `difflib.SequenceMatcher`. Matching spans are retained. For replacements it chooses the hint span when scene confidence is at least crop confidence, otherwise the crop span. Insertions from the crop are retained; deletions from the crop are retained only when scene confidence is higher. If the merge does not resemble a supported registration string, it returns the stronger complete read. The method name is `rapidocr-scene+crop` when a changed merged result is used.

### Optional Tesseract

Tesseract is used only if the `tesseract` executable is on `PATH` and the Python `pytesseract` import succeeds. Neither the executable nor binding is installed by the active `requirements.txt`; the binding line is commented as optional.

The reader tries `gray`, `otsu`, and `adaptive` variants with page segmentation modes 7 and 6, OCR engine mode 3, and whitelist `A-Z0-9`. It concatenates recognized words after removing non-alphanumerics. Per-word confidences are averaged. A matching Indian-format string gets a 0.35 ranking bonus; the resulting score is capped at 1 and is returned as the Tesseract confidence. This is an OCR alternative, not a plate detector.

### OpenCV template fallback

If RapidOCR does not take an early return, `_template_read` is also evaluated (alongside Tesseract if usable). It:

- uses inverse Otsu/adaptive masks;
- removes thin horizontal structures with a horizontal morphological opening whose width is `max(15, mask_width//4)` and height 1;
- clears 6% of the top/bottom and 2.5% of the left/right mask borders;
- retains connected components with height 20–94% of mask height, width 0.8–30% of mask width, and area at least 5;
- attempts vertical-projection valley splits for very wide components;
- proceeds only with 6–12 glyph components, groups them into up to two rows, and orders glyphs top-to-bottom then left-to-right;
- compares glyph masks against cached templates rendered with OpenCV Hershey Simplex, Duplex, Complex, and Plain fonts, thickness 1–3, on 64x48 canvases;
- scores templates with intersection-over-union and tries legal Indian state/district/series/serial length splits.

Template confidence is capped at 0.82. If no components or plausible segmentation exist, it returns empty text and zero confidence. It is a best-effort fallback; it is not a trained character model.

### Choosing among crop OCR methods

If neither RapidOCR early-return condition applies, the crop reader considers available RapidOCR, Tesseract, and template results. It ranks by confidence plus 0.12 for an Indian-format match. The pipeline then applies its own candidate score, registration-like preference, and early-stop rule described in Section 4.

## 8. Indian License Plate Handling

The implemented Indian syntax is the regular expression:

```text
^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$
```

This checks only the broad sequence of two letters, one or two digits, zero to three letters, and one to four digits. It does not validate state codes against a list, district codes, issuance, or whether a registration is real.

OCR normalization uppercases text and removes every character outside `A-Z0-9`, so separators such as spaces and hyphens are not preserved. For strings of length 7–11 only, `_correct_confusions` maps `0/1/2/5/8` to `O/I/Z/S/B` in the first two positions; maps `O/Q/D/I/L/Z/S/B/G` to digits `0/0/0/1/1/2/5/8/6` at the third position and in up to four trailing positions; and maps the fourth character only if it is already a digit. These are heuristic OCR substitutions, not validation.

The separate `looks_like_registration` predicate accepts either the Indian regex or the generic regex `[A-Z]{2,3}\d{3,4}`. The latter is used for candidate filtering/ranking and allows some non-Indian plates to be returned. `valid_indian_format` reports only the Indian regex result.

## 9. Outputs

For a successful run with an output directory, the pipeline writes:

| File | Contents |
|---|---|
| `01_input.jpg` | Original decoded input image, saved as JPEG. |
| `02_gray_clahe.png` | CLAHE grayscale localization image. |
| `02_gradient_x.png` | Absolute horizontal Sobel response. |
| `02_gradient_candidates.png` | Sobel binary map after morphology. |
| `02_edges.png` | Canny edge map before closing/dilation. |
| `02_edge_candidates.png` | Canny map after closing and dilation. |
| `02_adaptive_threshold.png` | Adaptive inverse threshold map before horizontal closing. |
| `03_candidate_proposals.jpg` | Working image with up to `max_candidates` ranked boxes and localization scores. |
| `04_plate_crop_native.png` | Selected plate extent crop at working-image resolution. |
| `04_plate_crop.png` | Same crop enlarged for inspection with cubic interpolation. |
| `05_plate_processed.png` | OCR's Otsu plate variant (or grayscale fallback). |
| `06_detected.jpg` | Original image with the selected box and final text overlaid. |
| `results.json` | Plate text, Indian-format flag, confidence, method, box, chosen proposal source, and evaluated candidate details. |

The two `04_...` files and `05_plate_processed.png` are omitted when no usable crop/processed image is returned. Internal variants such as enlarged grayscale, adaptive binary, and inverse masks are not separately saved. If the input was downscaled for processing, diagnostic images and crops use the working resolution; boxes in `results.json` are converted back to source-image coordinates.

The CLI also prints a JSON summary to standard output. `--show` opens an OpenCV window; it does not create another output file.

## 10. Test Result: `car01.webp`

The current source was run with:

```powershell
python -m alpr car01.webp --output alpr_output_car01_note
```

The actual output was:

- **Detected box:** `(220, 180, 51, 16)` in `(x, y, width, height)` order.
- **Recognized number:** `RJ25CA6667`.
- **Recognition method:** `rapidocr-scene+crop`.
- **`valid_indian_format`:** `true`.
- **Reported confidence:** `0.8845`.
- **Selected proposal source:** `ocr-text`.
- **Candidate localization score:** `0.9372`; crop OCR confidence: `0.8463`; candidate `combined_score`: `1.0` (capped after the registration-like ranking bonus).

The corresponding run artifacts are in `alpr_output_car01_note/`, including all maps above, the selected crop, annotation, and `results.json`. This is the current run's recorded result, not a manually transcribed value.

## 11. Project Structure

| Path | Responsibility |
|---|---|
| `alpr/localization.py` | Candidate dataclass; geometry/scoring; Sobel, Canny, and adaptive-threshold maps; contours; glyph grouping; NMS; crop margins. |
| `alpr/recognition.py` | Indian/generic text rules; scene OCR proposals; scene/crop text reconciliation; plate rectification and OCR preprocessing; RapidOCR; optional Tesseract; OpenCV template generation and fallback. |
| `alpr/pipeline.py` | Public `recognize_image` flow, image resize, proposal combination, multi-candidate evaluation, score/box conversion, result dataclass, annotation, and artifact writing. |
| `alpr/__main__.py` | CLI parsing, JSON printing, and optional OpenCV display window. |
| `alpr/__init__.py` | Exports `ALPRResult` and `recognize_image`. |
| `requirements.txt` | NumPy, OpenCV, and RapidOCR runtime requirements; Tesseract binding is commented as optional. |
| `README.md` | User-facing installation, run instructions, approach summary, examples, and limitations. |

## 12. Parameters

| Area | Parameter | Value |
|---|---|---|
| Input | Maximum image dimension | 1600 px; downscale uses `INTER_AREA`. |
| Localization | CLAHE | clip 2.2, tile 8x8. |
| Localization | Bilateral filter | d=7, color sigma=55, space sigma=55. |
| Sobel | Derivative | x order 1, y order 0, kernel 3; Otsu binary threshold. |
| Morphology | Horizontal kernel | width `max(9, int(0.045W)|1)`; height `max(3, int(0.012H)|1)`. |
| Morphology | Iterations | Sobel close 2, Sobel open 1 with 3x3; Canny close 2 and 3x3 dilate 1; adaptive close 1; bright fallback close 2. |
| Canny | Localization thresholds | 55/165; candidate ROI score uses 60/150. |
| Adaptive threshold | Main contour map | Gaussian, inverse, block 21, C=7. |
| Glyph grouping | Dynamic block | odd `max(15, int(0.065*min(H,W))|1)`, clamped to shorter image side; Gaussian inverse, C=5. |
| Geometry | Candidate aspect limits | 0.68–7.5; one-line prior centered at 4.0; two-line prior centered at 1.28. |
| Candidate filtering | NMS | IoU threshold 0.68; up to 30 classical candidates. |
| Pipeline | OCR candidate limit | 15 by default; CLI `--max-candidates` changes it. |
| Candidate score | Weights | aspect .27, brightness .27, contrast .16, components .18, rectangularity .07, edges .05. |
| Full-scene text | Candidate score | OCR .55, shape .25, constant .20; generic regex plus geometry filters. |
| Final rank | Weights/bonus | localization .42, OCR .58, registration-like bonus .15, rank score capped at 1. |
| Early stop | OCR confidence | Stop when a registration-like candidate has OCR confidence at least .72. |
| Rectification | Canny/contour | blur 3x3, Canny 35/120, close 3x3 twice, polygon epsilon .025 perimeter, area ≥38% crop. |
| Crop preprocessing | CLAHE/filter | clip 2.5, tile 4x4; bilateral 5/28/28; unsharp 1.7 and -0.7 at Gaussian sigma 1.1. |
| Crop preprocessing | Resize/padding | target height 112 if input height <32 else 96; scale 2–8; cubic; replicated border 8 vertical/10 horizontal. |
| Crop thresholding | Otsu/adaptive | Otsu binary; Gaussian adaptive binary block 31, C=9. |
| Dependencies | Requirements | `numpy>=1.24`, `opencv-python>=4.8`, `rapidocr-onnxruntime>=1.4.0`; optional Tesseract is not active in requirements. |

## 13. Error Handling and Fallbacks

- **Image cannot be loaded:** a path that `cv2.imread` cannot decode raises `FileNotFoundError`. An empty array raises `ValueError`. There is no custom recovery beyond these exceptions.
- **No plate candidate:** if the classical locator returns none, it tries its bright-region contour fallback. The pipeline still runs full-image OCR if RapidOCR is available. If there are no final candidates, it returns an empty plate string, false Indian-format flag, confidence 0, method `no-candidate`, and no box. It writes input/diagnostic/proposal/annotation/JSON files; crop and processed images are skipped.
- **RapidOCR unavailable or errors:** full-scene proposal detection catches exceptions and returns no text proposals. Crop recognition continues to Tesseract if available and then the OpenCV template method. RapidOCR is the required dependency in the normal installation, but this fallback behavior exists in code.
- **Tesseract unavailable:** `_tesseract_read` returns `None` if the executable is not on `PATH` or `pytesseract` cannot be imported. The template reader remains available. Tesseract was not the method used for the recorded `car01.webp` result.
- **Recognition returns empty/low confidence:** candidate evaluation continues; the selected box can still be returned with `plate_number` empty if final OCR confidence is below 0.12. The annotation then uses `plate detected` as its label.
- **Multiple candidates:** they are ranked, cropped, and read in descending proposal score up to the configured limit. A strong registration-like OCR result can stop the loop early. Evaluated candidates are then sorted by registration-like status and combined score. The system does not guarantee the top geometric proposal is the true plate.
- **Output-write errors:** no special write-error handler is implemented; filesystem/OpenCV errors may propagate or surface as failed `imwrite` calls.

## 14. Limitations

- The localization score is a heuristic built from rectangular geometry, brightness, contrast, components, and edges. It is not a learned plate detector and may prefer signs, grilles, reflections, or other text.
- Full-image RapidOCR can propose text from signs as well as plates. A registration-shaped regex reduces those proposals but does not prove a region is a plate.
- The RapidOCR scene/crop merge uses string alignment and confidence-based replacement. OCR confidences from the scene and crop are not calibrated to each other; merging can still preserve a wrong character.
- The accepted Indian regex is broad and does not check official state/district registries or issuance. The generic 2–3 letter + 3–4 digit rule is not specific to India.
- The character-template fallback uses a small set of OpenCV fonts and is substantially less reliable than a trained OCR model. It also imposes Indian-form splits.
- The code uses axis-aligned boxes for many proposals. Perspective correction only occurs when a sufficiently large four-sided contour is found inside a candidate crop; otherwise the crop stays unrectified.
- The current `car01.webp` test is one small, front-facing plate image. Other supplied runs in the workspace show different lighting/distance and include a non-Indian-format string, but they do not constitute a broad benchmark or validation of every two-line layout.
- Small, blurred, occluded, reflective, dirty, nonstandard, or steeply angled plates may be missed or misread. Detection confidence is heuristic, not a probability of correctness.

## 15. DSP / Image-Processing Contribution

**DSP and spatial image processing actually used:** grayscale conversion; CLAHE histogram redistribution; bilateral and Gaussian filtering; Sobel spatial derivative; Canny edge extraction; Otsu and adaptive thresholding; morphological close/open/dilation with rectangular structuring elements; connected-component statistics; cubic resampling; unsharp masking; and perspective warping. These operations alter or analyze image intensity and spatial structure before OCR.

**General computer vision:** contour extraction, bounding rectangles, aspect/area constraints, connected-component grouping, polygon approximation, min-area rectangles, non-maximum suppression, and candidate ranking are geometric image-analysis operations.

**OCR/classification:** RapidOCR is a general learned text detector/recognizer executed through ONNX Runtime. Optional Tesseract is another general OCR engine. The fallback compares segmented components with OpenCV-rendered glyph templates using mask IoU. Registration regexes and confusion maps are symbolic post-processing. These are not DSP filtering operations.

## 16. Important Design Decisions

1. **Use multiple classical localization cues.** Horizontal gradients, Canny, adaptive threshold contours, and component grouping respond differently to plate borders, text strokes, lighting, and blur; their proposals are scored together.
2. **Support several plate shapes.** The aspect prior explicitly includes broad single-line regions and near-square/taller two-row proposals; row pairing is implemented for two-line candidate generation and the fallback orders two OCR rows.
3. **Add scene-level OCR proposals after difficult examples.** Small/dark plates may not produce adequate binary connected components. Full-image general OCR can supply a text polygon and hint, while crop-level OCR rechecks it. This improves small-text proposals but also admits signs and non-plate text.
4. **Use crop-based enhancement before OCR.** CLAHE, bilateral filtering, sharpening, upscaling, and multiple threshold variants produce several views without changing the localization to a plate-specific neural detector.
5. **Use syntax only as a ranking aid.** Indian syntax helps resolve OCR ambiguities and rank results, while `looks_like_registration` also permits a common generic letter/digit pattern. Neither rule establishes legal validity.
6. **Save inspectable stages.** The output folder makes the candidate maps, selected crop, processed plate, annotated image, and scores available for review rather than returning only a string.

## Presentation-Relevant Facts

- **Problem:** automatically locate and read a vehicle registration plate from a vehicle photo.
- **Approach:** classical OpenCV image processing proposes and scores plate regions; general RapidOCR supplies full-scene text proposals and reads enhanced crops; Tesseract and an OpenCV template classifier are fallbacks.
- **Major stages:** grayscale/CLAHE/filtering → Sobel/Canny/adaptive-threshold and aligned-component proposals → candidate scoring/NMS → OCR text proposals → crop/conditional perspective warp → denoise/sharpen/upscale/threshold variants → OCR/merge/ranking → annotated image and JSON.
- **Recorded `car01.webp` output:** `RJ25CA6667`; box `(220, 180, 51, 16)`; method `rapidocr-scene+crop`; confidence `0.8845`; Indian regex check true.
- **Important caveat:** the confidence is heuristic. The Indian regex is not official validation; full-scene OCR can mistake signs for plates, and small/blurred or unusual plates can fail. One successful image is not a robustness benchmark.
