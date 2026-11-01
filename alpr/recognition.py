"""Plate cleanup and OCR with optional Tesseract and a local template fallback."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
from functools import lru_cache
from typing import Any

import cv2
import numpy as np


INDIA_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")
ALPHANUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RAPID_OCR_ENGINE = None


@dataclass
class OCRResult:
    text: str
    confidence: float
    method: str
    processed: np.ndarray
    variants: dict[str, np.ndarray]
    diagnostics: dict[str, object] | None = None


def rectify_plate(crop: np.ndarray) -> np.ndarray:
    """Try to rectify a plate quadrilateral found inside the candidate crop."""
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    possible: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        area = abs(cv2.contourArea(polygon))
        if len(polygon) != 4 or area < 0.38 * w * h:
            continue
        rect = cv2.minAreaRect(polygon)
        rw, rh = rect[1]
        if min(rw, rh) < 2:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not 0.65 <= ratio <= 7.8:
            continue
        # Prefer a large quadrilateral close to the crop bounds. This avoids
        # rectifying an interior grille slat as if it were the plate border.
        coverage = area / max(w * h, 1)
        bound = cv2.boundingRect(polygon)
        fill = area / max(bound[2] * bound[3], 1)
        quality = coverage * (0.65 + 0.35 * fill)
        possible.append((quality, cv2.boxPoints(rect)))

    if not possible:
        return crop
    possible.sort(key=lambda item: item[0], reverse=True)
    points = possible[0][1].astype(np.float32)
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
    ordered = np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)], points[np.argmax(sums)], points[np.argmax(diffs)]],
        dtype=np.float32,
    )
    tl, tr, br, bl = ordered
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < out_h and w / max(h, 1) > 2.1:
        # Correct a rotated single-line proposal while preserving tall,
        # two-line layouts in their original orientation.
        out_w, out_h = out_h, out_w
        ordered = ordered[[1, 2, 3, 0]]
    if out_w < 12 or out_h < 8:
        return crop
    target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(crop, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_plate(crop: np.ndarray) -> dict[str, np.ndarray]:
    """Normalize contrast, upscale, and produce threshold variants for OCR."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    if gray.size == 0:
        return {"gray": gray}
    # A light unsharp mask restores edges softened by a small source image.
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    smooth = cv2.bilateralFilter(gray, 5, 28, 28)
    sharpened = cv2.addWeighted(smooth, 1.7, cv2.GaussianBlur(smooth, (0, 0), 1.1), -0.7, 0)
    target_h = 112 if gray.shape[0] < 32 else 96
    scale = max(2.0, min(8.0, target_h / max(gray.shape[0], 1)))
    size = (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale)))
    enlarged = cv2.resize(sharpened, size, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.copyMakeBorder(enlarged, 8, 8, 10, 10, cv2.BORDER_REPLICATE)
    _, otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    # Inverse masks are useful for component segmentation; standard black text
    # on white remains the preferred input to Tesseract.
    return {
        "gray": enlarged,
        "otsu": otsu,
        "adaptive": adaptive,
        "otsu_inverse": cv2.bitwise_not(otsu),
        "adaptive_inverse": cv2.bitwise_not(adaptive),
    }


def _clean_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def looks_like_registration(value: str) -> bool:
    """Accept Indian numbers and common three-letter/four-digit plates."""
    text = _correct_confusions(value)
    return bool(
        INDIA_PATTERN.fullmatch(text)
        or re.fullmatch(r"[A-Z]{2,3}\d{3,4}", text)
    )


def detect_scene_text(image: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Return likely registration text boxes detected across the full photo.

    This supplements the classical candidate finder for small plates. Generic
    road-sign text is discarded using registration-shaped letter/digit runs.
    """
    global _RAPID_OCR_ENGINE
    try:
        if _RAPID_OCR_ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_ENGINE = RapidOCR()
        detected, _ = _RAPID_OCR_ENGINE(image)
    except Exception:
        return []
    if not detected:
        return []

    image_h, image_w = image.shape[:2]
    results: list[tuple[tuple[int, int, int, int], str, float]] = []
    for item in detected:
        try:
            polygon = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
            raw_text = str(item[1])
            confidence = float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        text = _correct_confusions(raw_text)
        if not looks_like_registration(text):
            continue
        x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
        x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w, x2), min(image_h, y2)
        w, h = x2 - x1, y2 - y1
        aspect = w / max(h, 1)
        if w < max(12, 0.025 * image_w) or h < max(6, 0.006 * min(image_h, image_w)):
            continue
        if not 1.45 <= aspect <= 9.0 or w * h > 0.12 * image_w * image_h:
            continue
        results.append(((int(x1), int(y1), int(w), int(h)), text, confidence))
    return results


def merge_text_reads(hint: str, crop_text: str, hint_confidence: float, crop_confidence: float) -> str:
    """Align scene-level and crop-level OCR, preserving consensus characters."""
    from difflib import SequenceMatcher

    a, b = _clean_text(hint), _clean_text(crop_text)
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a
    if a in b:
        return b
    if b in a:
        return a

    merged: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            merged.append(a[i1:i2])
        elif tag == "replace":
            merged.append(a[i1:i2] if hint_confidence >= crop_confidence else b[j1:j2])
        elif tag == "delete":
            if hint_confidence >= crop_confidence:
                merged.append(a[i1:i2])
        elif tag == "insert":
            merged.append(b[j1:j2])
    value = "".join(merged)
    if looks_like_registration(value):
        return value
    # Keep the stronger complete OCR result if a merged string breaks the
    # registration-shaped pattern.
    return a if hint_confidence >= crop_confidence else b


def _correct_confusions(value: str) -> str:
    """Apply common OCR substitutions only when the Indian plate shape allows."""
    text = _clean_text(value)
    if len(text) < 7 or len(text) > 11:
        return text
    letters = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
    digits = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
    chars = list(text)
    for i in range(min(2, len(chars))):
        chars[i] = letters.get(chars[i], chars[i])
    if len(chars) > 2:
        chars[2] = digits.get(chars[2], chars[2])
    if len(chars) > 3 and chars[3].isdigit():
        chars[3] = digits.get(chars[3], chars[3])
    # The trailing one to four symbols are serial digits.
    trailing_digits = 0
    for symbol in reversed(chars):
        mapped = digits.get(symbol, symbol)
        if mapped.isdigit() and trailing_digits < 4:
            trailing_digits += 1
        else:
            break
    for i in range(len(chars) - trailing_digits, len(chars)):
        chars[i] = digits.get(chars[i], chars[i])
    return "".join(chars)


def _indian_format_score(text: str) -> float:
    cleaned = _correct_confusions(text)
    if INDIA_PATTERN.fullmatch(cleaned):
        return 1.0
    return 0.0


def _tesseract_read(variants: dict[str, np.ndarray]) -> tuple[str, float, str] | None:
    if not shutil.which("tesseract"):
        return None
    try:
        import pytesseract
    except ImportError:
        return None
    choices: list[tuple[float, str, str]] = []
    for name in ("gray", "otsu", "adaptive"):
        arr = variants.get(name)
        if arr is None:
            continue
        # PSM 7 is suited to a one-line registration. PSM 6 can combine the
        # two rows of a tall Indian plate.
        for psm in (7, 6):
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            try:
                data: dict[str, Any] = pytesseract.image_to_data(
                    arr, config=config, output_type=pytesseract.Output.DICT
                )
            except Exception:
                continue
            pieces: list[str] = []
            confidences: list[float] = []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                word = _clean_text(word)
                try:
                    score = float(conf)
                except (TypeError, ValueError):
                    score = -1.0
                if word:
                    pieces.append(word)
                    if score >= 0:
                        confidences.append(score / 100.0)
            raw = "".join(pieces)
            text = _correct_confusions(raw)
            conf = float(np.mean(confidences)) if confidences else 0.0
            format_bonus = _indian_format_score(text)
            choices.append((conf + 0.35 * format_bonus, text, f"tesseract-psm{psm}-{name}"))
    if not choices:
        return None
    choices.sort(reverse=True)
    rank, text, method = choices[0]
    return text, min(1.0, max(0.0, rank)), method


def _rapidocr_read(variants: dict[str, np.ndarray]) -> tuple[float, str, str] | None:
    """Read the enhanced plate with RapidOCR's small ONNX text recognizer."""
    global _RAPID_OCR_ENGINE
    try:
        if _RAPID_OCR_ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_ENGINE = RapidOCR()
    except (ImportError, OSError, RuntimeError):
        return None

    choices: list[tuple[float, float, str, str]] = []
    for name in ("gray", "otsu", "adaptive"):
        arr = variants.get(name)
        if arr is None or arr.size == 0:
            continue
        try:
            detected, _ = _RAPID_OCR_ENGINE(arr)
        except Exception:
            continue
        if not detected:
            continue
        # The plate has already been localized, so stitch OCR text regions in
        # row-major order instead of requiring a second text detector to span
        # its full width.
        detected = sorted(detected, key=lambda item: (float(np.mean(np.asarray(item[0])[:, 1])), float(np.mean(np.asarray(item[0])[:, 0]))))
        raw = "".join(str(item[1]) for item in detected)
        text = _correct_confusions(raw)
        scores = [float(item[2]) for item in detected if len(item) > 2]
        confidence = float(np.mean(scores)) if scores else 0.0
        bonus = 0.18 if INDIA_PATTERN.fullmatch(text) else 0.0
        choices.append((confidence + bonus, confidence, text, f"rapidocr-{name}"))
    if not choices:
        return None
    choices.sort(reverse=True)
    _, confidence, text, method = choices[0]
    return min(1.0, max(0.0, confidence)), text, method


@lru_cache(maxsize=1)
def _templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {}
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN]
    for symbol in ALPHANUM:
        items: list[np.ndarray] = []
        for font in fonts:
            for thickness in (1, 2, 3):
                canvas = np.zeros((64, 48), dtype=np.uint8)
                scale = 1.15 if font != cv2.FONT_HERSHEY_PLAIN else 1.9
                (tw, th), baseline = cv2.getTextSize(symbol, font, scale, thickness)
                x = max(0, (canvas.shape[1] - tw) // 2)
                y = max(th + 2, (canvas.shape[0] + th - baseline) // 2)
                cv2.putText(canvas, symbol, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)
                items.append(canvas)
        templates[symbol] = items
    return templates


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    h, w = glyph.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((64, 48), np.uint8)
    # Keep the glyph's natural aspect while centering it in the template cell.
    target_w, target_h = 34, 48
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(glyph, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((64, 48), np.uint8)
    x, y = (48 - resized.shape[1]) // 2, (64 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _glyph_scores(glyph: np.ndarray, template_set: dict[str, list[np.ndarray]]) -> dict[str, float]:
    normalized = _normalize_glyph(glyph)
    scores: dict[str, float] = {}
    for char, templates in template_set.items():
        best = 0.0
        for template in templates:
            intersection = float(np.count_nonzero((normalized > 0) & (template > 0)))
            union = float(np.count_nonzero((normalized > 0) | (template > 0)))
            best = max(best, intersection / union if union else 0.0)
        scores[char] = best
    return scores


def _split_touching(binary: np.ndarray, x1: int, x2: int, y1: int, y2: int) -> list[np.ndarray]:
    piece = binary[y1:y2, x1:x2]
    width = piece.shape[1]
    if width <= 1:
        return [piece]
    projection = np.count_nonzero(piece, axis=0)
    if width / max(piece.shape[0], 1) < 0.95 or projection.max(initial=0) == 0:
        return [piece]
    # Use a deep local valley to split glyphs that blurred into one component.
    smoothed = np.convolve(projection.astype(np.float32), np.ones(3) / 3, mode="same")
    threshold = max(1.0, float(smoothed.max()) * 0.12)
    cuts: list[int] = []
    in_gap = smoothed[0] <= threshold
    gap_start = 0
    for i in range(1, width):
        gap = smoothed[i] <= threshold
        if gap and not in_gap:
            gap_start = i
        elif not gap and in_gap:
            if i - gap_start >= max(1, int(0.025 * width)):
                cuts.append((gap_start + i) // 2)
        in_gap = gap
    bounds = [0] + cuts + [width]
    if len(bounds) > 2:
        return [piece[:, a:b] for a, b in zip(bounds, bounds[1:]) if b - a >= 2]
    return [piece]


def _template_read(
    variants: dict[str, np.ndarray],
    diagnostics: dict[str, object] | None = None,
) -> tuple[str, float, str]:
    template_set = _templates()
    proposals: list[tuple[float, str, str]] = []
    variant_diagnostics: list[dict[str, object]] | None = [] if diagnostics is not None else None
    for variant_name in ("otsu_inverse", "adaptive_inverse"):
        mask = variants.get(variant_name)
        if mask is None or mask.size == 0:
            if variant_diagnostics is not None:
                variant_diagnostics.append({"variant": variant_name, "status": "unavailable"})
            continue
        variant_info: dict[str, object] | None = {"variant": variant_name} if diagnostics is not None else None
        # Open thin horizontal plate borders away before finding glyphs.
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, mask.shape[1] // 4), 1)))
        cleaned = cv2.subtract(mask, horizontal)
        cleaned[: max(1, int(cleaned.shape[0] * 0.06)), :] = 0
        cleaned[-max(1, int(cleaned.shape[0] * 0.06)) :, :] = 0
        cleaned[:, : max(1, int(cleaned.shape[1] * 0.025))] = 0
        cleaned[:, -max(1, int(cleaned.shape[1] * 0.025)) :] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        h, w = cleaned.shape
        comps: list[tuple[int, int, int, int, np.ndarray]] = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if ch < 0.20 * h or ch > 0.94 * h or cw < max(2, 0.008 * w) or cw > 0.30 * w or area < 5:
                continue
            glyph = np.where(labels[y : y + ch, x : x + cw] == i, 255, 0).astype(np.uint8)
            for piece in _split_touching(glyph, 0, cw, 0, ch):
                if piece.shape[1] >= 2:
                    comps.append((x, y, piece.shape[1], piece.shape[0], piece))
        if variant_info is not None:
            variant_info["connected_component_count"] = int(n - 1)
            variant_info["character_component_count"] = len(comps)
        if not 6 <= len(comps) <= 12:
            if variant_info is not None:
                variant_info["status"] = "component_count_out_of_range"
                variant_diagnostics.append(variant_info)
            continue
        # A two-line layout has clear vertical clusters. Keep character order
        # top-to-bottom, then left-to-right within each row.
        centers = np.array([y + ch / 2 for _, y, _, ch, _ in comps])
        order = np.argsort(centers)
        groups: list[list[int]] = []
        for idx in order:
            cy = centers[idx]
            if not groups or abs(cy - np.mean([centers[j] for j in groups[-1]])) > 0.24 * h:
                groups.append([int(idx)])
            else:
                groups[-1].append(int(idx))
        if len(groups) > 2:
            groups = sorted(groups, key=len, reverse=True)[:2]
        sequence: list[int] = []
        for group in sorted(groups, key=lambda g: np.mean([centers[j] for j in g])):
            sequence.extend(sorted(group, key=lambda j: comps[j][0]))
        char_scores = [_glyph_scores(comps[i][4], template_set) for i in sequence]
        count = len(char_scores)
        if variant_info is not None:
            variant_info["row_count"] = len(groups)
            variant_info["row_glyph_counts"] = [len(group) for group in groups]

        # Evaluate all valid Indian state/district/series/serial splits. The
        # format prior resolves ambiguous shapes such as 0/O and 1/I.
        best_text, best_score = "", -1.0
        best_layout: dict[str, object] | None = None
        for district_len in (1, 2):
            for series_len in range(0, 4):
                serial_len = count - 2 - district_len - series_len
                if not 1 <= serial_len <= 4:
                    continue
                expected: list[str] = ["L", "L"] + ["D"] * district_len + ["L"] * series_len + ["D"] * serial_len
                if len(expected) != count:
                    continue
                chars: list[str] = []
                matched: list[float] = []
                for i, kind in enumerate(expected):
                    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if kind == "L" else "0123456789"
                    symbol = max(allowed, key=lambda ch: char_scores[i].get(ch, 0.0))
                    chars.append(symbol)
                    matched.append(char_scores[i].get(symbol, 0.0))
                mean_score = float(np.mean(matched))
                # Slight preference for the common two-digit district and
                # four-digit serial without forcing that layout.
                prior = 0.025 * (district_len == 2) + 0.02 * (serial_len == 4)
                score = mean_score + prior
                if score > best_score:
                    best_score, best_text = score, "".join(chars)
                    if variant_info is not None:
                        character_matches = []
                        for index, kind in enumerate(expected):
                            allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if kind == "L" else "0123456789"
                            alternatives = sorted(
                                (
                                    (symbol, float(value))
                                    for symbol, value in char_scores[index].items()
                                    if symbol in allowed
                                ),
                                key=lambda item: item[1],
                                reverse=True,
                            )[:3]
                            character_matches.append(
                                {
                                    "position": index,
                                    "expected_type": "letter" if kind == "L" else "digit",
                                    "selected": chars[index],
                                    "selected_match_score": float(matched[index]),
                                    "top_alternatives": [
                                        {"character": symbol, "match_score": value}
                                        for symbol, value in alternatives
                                    ],
                                }
                            )
                        best_layout = {
                            "district_digit_count": district_len,
                            "series_letter_count": series_len,
                            "serial_digit_count": serial_len,
                            "mean_glyph_match_score": mean_score,
                            "layout_prior": float(prior),
                            "character_matches": character_matches,
                        }
        if best_text:
            proposals.append((best_score, best_text, f"opencv-templates-{variant_name}"))
            if variant_info is not None:
                variant_info["status"] = "matched"
                variant_info["best_text"] = best_text
                variant_info["best_score"] = float(best_score)
                variant_info["best_layout"] = best_layout
        elif variant_info is not None:
            variant_info["status"] = "no_valid_layout"
        if variant_info is not None:
            variant_diagnostics.append(variant_info)
    if not proposals:
        if diagnostics is not None:
            diagnostics.update({"variants": variant_diagnostics or [], "selected": None})
        return "", 0.0, "opencv-templates-no-segments"
    proposals.sort(reverse=True)
    selected_text = proposals[0][1]
    selected_score = min(0.82, max(0.0, proposals[0][0]))
    selected_method = proposals[0][2]
    if diagnostics is not None:
        diagnostics.update(
            {
                "variants": variant_diagnostics or [],
                "competing_variants": [
                    {"method": method, "text": text, "score": float(score)}
                    for score, text, method in proposals
                ],
                "selected": {
                    "method": selected_method,
                    "text": selected_text,
                    "score": float(selected_score),
                },
            }
        )
    return selected_text, selected_score, selected_method


def recognize_crop(crop: np.ndarray, include_diagnostics: bool = False) -> OCRResult:
    """Recognize a plate crop with RapidOCR, Tesseract, or local templates."""
    rectified = rectify_plate(crop)
    variants = preprocess_plate(rectified)
    rapid = _rapidocr_read(variants)
    if rapid and rapid[0] >= 0.75 and INDIA_PATTERN.fullmatch(rapid[1]):
        processed = variants.get("otsu", variants.get("gray"))
        diagnostics = (
            {
                "selection_path": "rapidocr_format_fast_path",
                "ocr_method_candidates": [
                    {"method": rapid[2], "text": rapid[1], "confidence": rapid[0]}
                ],
                "template_evidence": {"evaluated": False, "reason": "RapidOCR fast path"},
            }
            if include_diagnostics
            else None
        )
        return OCRResult(rapid[1], rapid[0], rapid[2], processed, variants, diagnostics)
    if rapid and rapid[0] >= 0.50 and not shutil.which("tesseract"):
        processed = variants.get("otsu", variants.get("gray"))
        diagnostics = (
            {
                "selection_path": "rapidocr_no_tesseract_fast_path",
                "ocr_method_candidates": [
                    {"method": rapid[2], "text": rapid[1], "confidence": rapid[0]}
                ],
                "template_evidence": {"evaluated": False, "reason": "RapidOCR fast path"},
            }
            if include_diagnostics
            else None
        )
        return OCRResult(rapid[1], rapid[0], rapid[2], processed, variants, diagnostics)
    tess = _tesseract_read(variants)
    template_diagnostics: dict[str, object] | None = {} if include_diagnostics else None
    fallback_text, fallback_conf, fallback_method = _template_read(variants, template_diagnostics)
    choices: list[tuple[float, str, str]] = []
    if rapid:
        choices.append(rapid)
    if tess and tess[0]:
        choices.append((tess[1], tess[0], tess[2]))
    if fallback_text:
        choices.append((fallback_conf, fallback_text, fallback_method))
    if choices:
        # Prefer syntactically plausible reads when model confidences are
        # close, but keep OCR confidence as the dominant signal.
        choices.sort(key=lambda item: item[0] + (0.12 if INDIA_PATTERN.fullmatch(item[1]) else 0.0), reverse=True)
        confidence, text, method = choices[0]
        confidence = min(1.0, confidence)
    else:
        confidence, text, method = 0.0, "", "no-ocr-result"
    processed = variants.get("otsu", variants.get("gray"))
    diagnostics = None
    if include_diagnostics:
        diagnostics = {
            "selection_path": "multi_method_rank",
            "ocr_method_candidates": [
                {
                    "method": candidate_method,
                    "text": candidate_text,
                    "confidence": float(candidate_confidence),
                    "format_bonus_for_method_selection": 0.12
                    if INDIA_PATTERN.fullmatch(candidate_text)
                    else 0.0,
                    "method_selection_score": float(
                        candidate_confidence
                        + (0.12 if INDIA_PATTERN.fullmatch(candidate_text) else 0.0)
                    ),
                }
                for candidate_confidence, candidate_text, candidate_method in choices
            ],
            "selected_method": method,
            "template_evidence": template_diagnostics,
        }
    return OCRResult(text, confidence, method, processed, variants, diagnostics)
