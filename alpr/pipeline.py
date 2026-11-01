"""End-to-end license plate localization, rectification, and recognition."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .localization import Candidate, crop_candidate, localize_plates
from .recognition import (
    INDIA_PATTERN,
    OCRResult,
    detect_scene_text,
    looks_like_registration,
    merge_text_reads,
    recognize_crop,
)


@dataclass
class ALPRResult:
    plate_number: str
    valid_indian_format: bool
    confidence: float
    ocr_method: str
    plate_box: tuple[int, int, int, int] | None
    plate_crop: np.ndarray | None
    processed_plate: np.ndarray | None
    diagnostics: dict[str, np.ndarray]
    candidates: list[dict]
    annotated: np.ndarray | None


def _save_image(path: Path, image: np.ndarray | None) -> None:
    if image is None or image.size == 0:
        return
    cv2.imwrite(str(path), image)


def _plate_extent(candidate: Candidate, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Expand a tight text-run box to include the surrounding plate body."""
    x, y, w, h = candidate.box
    image_h, image_w = image_shape[:2]
    if candidate.source == "aligned-glyphs":
        if w / max(h, 1) > 2.1:
            pad_x, pad_y = max(2, int(round(w * 0.07))), max(3, int(round(h * 0.24)))
        else:
            pad_x, pad_y = max(3, int(round(w * 0.10))), max(2, int(round(h * 0.10)))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(image_w, x + w + pad_x), min(image_h, y + h + pad_y)
        return x1, y1, x2 - x1, y2 - y1
    if candidate.source == "ocr-text":
        pad_x, pad_y = max(3, int(round(w * 0.12))), max(2, int(round(h * 0.12)))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(image_w, x + w + pad_x), min(image_h, y + h + pad_y)
        return x1, y1, x2 - x1, y2 - y1
    return candidate.box


def recognize_image(
    image_or_path: np.ndarray | str | Path,
    output_dir: str | Path | None = None,
    max_candidates: int = 15,
    include_diagnostics: bool = True,
) -> ALPRResult:
    """Run the full ALPR pipeline on a photo or an already loaded BGR image.

    If ``output_dir`` is supplied, the method saves each major processing
    stage along with a JSON summary for inspection. Set ``include_diagnostics``
    to false to avoid retaining or writing diagnostic images.
    """
    if isinstance(image_or_path, (str, Path)):
        original = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_or_path}")
    else:
        original = image_or_path.copy()
        if original.ndim == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if original is None or original.size == 0:
        raise ValueError("Input image is empty")

    original_h, original_w = original.shape[:2]
    scale = min(1.0, 1600.0 / max(original_h, original_w))
    working = original if scale == 1.0 else cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    candidates, diagnostics = localize_plates(working, include_diagnostics=include_diagnostics)
    # Full-scene OCR supplies text-region proposals for small plates whose
    # glyph components disappear in a dark or distant photograph. Classical
    # edge/contrast proposals remain in the candidate pool as well.
    for text_box, text_hint, text_confidence in detect_scene_text(working):
        _, _, text_w, text_h = text_box
        aspect = text_w / max(text_h, 1)
        shape_score = float(np.exp(-abs(np.log(aspect / 4.0)) / 1.5))
        candidate_score = 0.55 * text_confidence + 0.25 * shape_score + 0.20
        text_features = (
            {
                "scene_text_hint_normalized": text_hint,
                "scene_text_confidence": float(text_confidence),
                "scene_text_aspect_ratio": float(aspect),
                "scene_text_shape_score": shape_score,
                "scene_text_candidate_score": float(candidate_score),
            }
            if include_diagnostics
            else None
        )
        candidates.append(
            Candidate(
                text_box,
                candidate_score,
                "ocr-text",
                text_hint,
                text_confidence,
                diagnostic_features=text_features,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)

    annotated_candidates = None
    if include_diagnostics:
        annotated_candidates = working.copy()
        for rank, candidate in enumerate(candidates[:max_candidates], 1):
            x, y, w, h = _plate_extent(candidate, working.shape)
            cv2.rectangle(annotated_candidates, (x, y), (x + w, y + h), (40, 200, 255), 1)
            cv2.putText(
                annotated_candidates,
                f"{rank}:{candidate.score:.2f}",
                (x, max(12, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (40, 100, 255),
                1,
                cv2.LINE_AA,
            )

    evaluated: list[tuple[float, Candidate, np.ndarray, OCRResult]] = []
    for candidate in candidates[:max_candidates]:
        crop = crop_candidate(working, candidate)
        if crop.size == 0:
            continue
        ocr = recognize_crop(crop, include_diagnostics=include_diagnostics)
        if candidate.text_hint:
            crop_ocr_text = ocr.text
            crop_ocr_confidence = ocr.confidence
            combined_text = merge_text_reads(
                candidate.text_hint,
                ocr.text,
                candidate.text_confidence,
                ocr.confidence,
            )
            if combined_text and combined_text != ocr.text:
                merged_confidence = 0.5 * (candidate.text_confidence + ocr.confidence)
                ocr_diagnostics = None
                if include_diagnostics:
                    ocr_diagnostics = dict(ocr.diagnostics or {})
                    ocr_diagnostics["scene_text_merge"] = {
                        "scene_text_hint_normalized": candidate.text_hint,
                        "scene_text_confidence": float(candidate.text_confidence),
                        "crop_ocr_text": crop_ocr_text,
                        "crop_ocr_confidence": float(crop_ocr_confidence),
                        "merged_text": combined_text,
                        "merged_confidence": float(merged_confidence),
                    }
                ocr = OCRResult(
                    combined_text,
                    merged_confidence,
                    "rapidocr-scene+crop",
                    ocr.processed,
                    ocr.variants,
                    ocr_diagnostics,
                )
        pattern_valid = bool(INDIA_PATTERN.fullmatch(ocr.text))
        registration_like = looks_like_registration(ocr.text)
        base_score = 0.42 * candidate.score + 0.58 * ocr.confidence
        pattern_bonus = 0.15 if registration_like else 0.0
        combined = min(1.0, base_score + pattern_bonus)
        evaluated.append((combined, candidate, crop, ocr))
        # A high-confidence, syntactically plausible read on the strongest
        # localization proposal is a useful stopping condition. Lower-ranked
        # background rectangles otherwise add latency and OCR false positives.
        if registration_like and ocr.confidence >= 0.72:
            break

    if evaluated:
        evaluated.sort(
            key=lambda item: (
                looks_like_registration(item[3].text),
                item[0],
            ),
            reverse=True,
        )
        combined, best, crop, ocr = evaluated[0]
        pattern_valid = bool(INDIA_PATTERN.fullmatch(ocr.text))
        # Convert proposal coordinates back to the supplied source image.
        best_box = _plate_extent(best, working.shape)
        x, y, w, h = best_box
        if scale != 1:
            box = tuple(int(round(v / scale)) for v in (x, y, w, h))
        else:
            box = best_box
        final_text = ocr.text if ocr.confidence >= 0.12 else ""
        final_confidence = 0.42 * best.score + 0.58 * ocr.confidence
        final_method = ocr.method
        final_crop = working[y : y + h, x : x + w].copy()
        final_processed = ocr.processed
        selected_candidate = best
    else:
        combined, best, crop, ocr = 0.0, None, None, None
        box, pattern_valid, final_text, final_confidence, final_method = None, False, "", 0.0, "no-candidate"
        final_crop, final_processed, selected_candidate = None, None, None

    annotated = original.copy() if include_diagnostics else None
    if annotated is not None and box:
        x, y, w, h = box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), max(2, original_w // 300))
        label = final_text or "plate detected"
        baseline_y = max(20, y - 7)
        cv2.putText(annotated, label, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 40), 2, cv2.LINE_AA)

    candidate_rows: list[dict] = []
    for rank, (score, candidate, _, candidate_ocr) in enumerate(evaluated, 1):
        x, y, w, h = _plate_extent(candidate, working.shape)
        row = {
            "rank": rank,
            "box": [int(round(v / scale)) for v in (x, y, w, h)],
            "localization_score": round(candidate.score, 4),
            "combined_score": round(score, 4),
            "text": candidate_ocr.text,
            "registration_like": looks_like_registration(candidate_ocr.text),
            "ocr_confidence": round(candidate_ocr.confidence, 4),
            "ocr_method": candidate_ocr.method,
            "proposal_source": candidate.source,
        }
        if include_diagnostics:
            raw_proposal_box = candidate.raw_box or candidate.box
            row["raw_proposal_box_xywh"] = [
                int(round(v / scale)) for v in raw_proposal_box
            ]
            row["scored_proposal_box_xywh"] = [
                int(round(v / scale)) for v in candidate.box
            ]
            row["localization_features"] = candidate.diagnostic_features
            row["ocr_evidence"] = candidate_ocr.diagnostics
            if candidate.text_hint:
                row["scene_text_hint_normalized"] = candidate.text_hint
        candidate_rows.append(row)

    selection_explanation = None
    if include_diagnostics:
        selection_explanation = {
            "ordering": ["registration_like: true first", "combined_score: descending"],
            "scope": "OCR-evaluated candidates only; proposals beyond max_candidates are not ranked here",
            "selected_rank": 1 if candidate_rows else None,
            "winner_comparisons": [],
        }
        if candidate_rows:
            winner = candidate_rows[0]
            for other in candidate_rows[1:]:
                if winner["registration_like"] != other["registration_like"]:
                    reason = (
                        "Winner is registration_like; that primary sort key precedes a non-registration-like candidate."
                    )
                elif winner["combined_score"] > other["combined_score"]:
                    reason = (
                        "Both candidates have the same registration_like value; winner has the higher combined_score."
                    )
                else:
                    reason = "The selection keys tie; stable sorting preserves the earlier candidate."
                selection_explanation["winner_comparisons"].append(
                    {"candidate_rank": other["rank"], "reason": reason}
                )

    result = ALPRResult(
        plate_number=final_text,
        valid_indian_format=pattern_valid,
        confidence=round(float(final_confidence), 4),
        ocr_method=final_method,
        plate_box=box,
        plate_crop=final_crop,
        processed_plate=final_processed,
        diagnostics=diagnostics,
        candidates=candidate_rows,
        annotated=annotated,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if include_diagnostics:
            _save_image(out / "01_input.jpg", original)
            for name, diagnostic in diagnostics.items():
                _save_image(out / f"02_{name}.png", diagnostic)
            _save_image(out / "03_candidate_proposals.jpg", annotated_candidates)
            if final_crop is not None and final_crop.size:
                _save_image(out / "04_plate_crop_native.png", final_crop)
                display_scale = max(4.0, min(10.0, 500.0 / max(final_crop.shape[1], 1)))
                display_crop = cv2.resize(
                    final_crop,
                    None,
                    fx=display_scale,
                    fy=display_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                _save_image(out / "04_plate_crop.png", display_crop)
            _save_image(out / "05_plate_processed.png", final_processed)
            _save_image(out / "06_detected.jpg", annotated)
        else:
            evidence = out / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            _save_image(evidence / "original.jpg", original)
        summary = {
            "plate_number": result.plate_number,
            "valid_indian_format": result.valid_indian_format,
            "confidence": result.confidence,
            "ocr_method": result.ocr_method,
            "plate_box_xywh": result.plate_box,
            "selected_proposal_source": selected_candidate.source if selected_candidate else None,
            "candidates": candidate_rows,
        }
        if include_diagnostics:
            summary["selection_explanation"] = selection_explanation
            summary["diagnostic_coordinate_notes"] = {
                "box_coordinates": "input-image pixels",
                "localization_pixel_features": "scaled working-image pixels",
                "working_image_scale": float(scale),
            }
        (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return result
