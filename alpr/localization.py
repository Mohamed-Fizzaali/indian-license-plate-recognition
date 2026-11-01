"""Plate localization using multi-scale, classical image processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Candidate:
    box: tuple[int, int, int, int]  # x, y, width, height
    score: float
    source: str
    text_hint: str = ""
    text_confidence: float = 0.0
    diagnostic_features: dict[str, object] | None = None
    raw_box: tuple[int, int, int, int] | None = None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _component_count(binary: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    h, w = binary.shape
    count = 0
    for x, y, cw, ch, area in stats[1:n]:
        if 0.18 * h <= ch <= 0.88 * h and 0.012 * w <= cw <= 0.24 * w and area >= 5:
            count += 1
    return count


def _score_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    contour_area: float,
    diagnostic_features: dict[str, object] | None = None,
) -> float:
    x, y, w, h = box
    image_h, image_w = gray.shape
    scale_dim = min(image_h, image_w)
    aspect = w / max(h, 1)

    # Indian plates are usually long and shallow, but two-line plates can be
    # nearly square. A broad prior keeps both layouts in the search.
    if not (0.68 <= aspect <= 7.5):
        return -1.0
    if w < max(18, 0.035 * image_w) or h < max(9, 0.012 * scale_dim):
        return -1.0
    if h > 0.46 * image_h or w * h > 0.30 * image_w * image_h:
        return -1.0

    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return -1.0
    ratio = w / h
    if ratio >= 2.1:
        aspect_score = np.exp(-abs(np.log(ratio / 4.0)) / 1.4)
    else:
        aspect_score = 0.82 * np.exp(-abs(np.log(ratio / 1.28)) / 0.9)

    light_fraction = float(np.mean(roi > 150))
    dark_fraction = float(np.mean(roi < 105))
    brightness_score = max(0.0, 1.0 - abs(light_fraction - 0.60) / 0.70)
    roi_std = float(np.std(roi))
    contrast_score = min(roi_std / 58.0, 1.0)

    # Otsu's inverse threshold approximates dark registration glyphs on a
    # reflective white plate. Component evidence is deliberately soft since
    # blur, glare, and two-line layouts can merge or fragment glyphs.
    smooth = cv2.GaussianBlur(roi, (3, 3), 0)
    _, ink = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    components = _component_count(ink)
    component_score = max(0.0, 1.0 - abs(components - 8) / 10.0)
    char_score = component_score
    if dark_fraction < 0.015:
        char_score *= 0.25

    rectangularity = min(contour_area / max(float(w * h), 1.0), 1.0)
    # Char proposals can have low rectangularity; cap the contribution.
    rect_score = 0.5 + 0.5 * rectangularity
    edge_mean = cv2.Canny(roi, 60, 150).mean()
    edge_score = min(edge_mean / 42.0, 1.0)

    if diagnostic_features is not None:
        diagnostic_features.update(
            {
                "aspect_ratio": float(aspect),
                "proposal_width_fraction": float(w / image_w),
                "proposal_height_fraction": float(h / image_h),
                "proposal_area_fraction": float((w * h) / (image_w * image_h)),
                "aspect_score": float(aspect_score),
                "light_fraction_gt_150": float(light_fraction),
                "brightness_score": float(brightness_score),
                "dark_fraction_lt_105": float(dark_fraction),
                "roi_stddev": roi_std,
                "contrast_score": float(contrast_score),
                "connected_component_count": int(components),
                "component_score_before_dark_penalty": float(component_score),
                "component_score": float(char_score),
                "contour_rectangularity": float(rectangularity),
                "rectangularity_score": float(rect_score),
                "edge_mean": float(edge_mean),
                "edge_score": float(edge_score),
            }
        )

    return float(
        0.27 * aspect_score
        + 0.27 * brightness_score
        + 0.16 * contrast_score
        + 0.18 * char_score
        + 0.07 * rect_score
        + 0.05 * edge_score
    )


def _group_glyph_candidates(
    gray: np.ndarray,
    include_diagnostics: bool = False,
) -> list[Candidate]:
    """Group similarly sized, aligned dark components into text-line proposals."""
    image_h, image_w = gray.shape
    scale_dim = min(image_h, image_w)
    block = max(15, int(min(image_h, image_w) * 0.065) | 1)
    block = min(block, min(image_h, image_w) - (1 - min(image_h, image_w) % 2))
    if block < 3:
        return []
    adaptive = cv2.adaptiveThreshold(
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        5,
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(adaptive, 8)
    glyphs: list[tuple[int, int, int, int]] = []
    for x, y, w, h, area in stats[1:n]:
        if (
            max(2, 0.0025 * image_w) <= w <= 0.075 * image_w
            and max(3, 0.008 * scale_dim) <= h <= 0.13 * image_h
            and area >= 5
            and 0.12 <= w / max(h, 1) <= 1.4
        ):
            glyphs.append((int(x), int(y), int(w), int(h)))
    glyphs.sort(key=lambda box: box[0])

    line_rows: list[tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int], float]] = []
    proposals: list[Candidate] = []
    seen_rows: set[tuple[int, int, int, int]] = set()
    # Build possible text rows around each seed glyph. The previous greedy
    # scan let unrelated objects between adjacent letters break a valid row.
    for seed in glyphs:
        seed_cy = seed[1] + seed[3] / 2
        row = [
            glyph
            for glyph in glyphs
            if abs((glyph[1] + glyph[3] / 2) - seed_cy) <= 0.48 * max(seed[3], glyph[3])
            and 0.58 <= glyph[3] / max(seed[3], 1) <= 1.72
        ]
        row.sort(key=lambda item: item[0])
        runs: list[list[tuple[int, int, int, int]]] = []
        current: list[tuple[int, int, int, int]] = []
        for glyph in row:
            if current:
                previous = current[-1]
                gap = glyph[0] - (previous[0] + previous[2])
                char_h = float(np.median([item[3] for item in current]))
                if gap > 1.35 * max(char_h, glyph[3]) or gap < -0.20 * char_h:
                    runs.append(current)
                    current = []
            current.append(glyph)
        if current:
            runs.append(current)

        for run in runs:
            if not 3 <= len(run) <= 13:
                continue
            x1 = min(item[0] for item in run)
            y1 = min(item[1] for item in run)
            x2 = max(item[0] + item[2] for item in run)
            y2 = max(item[1] + item[3] for item in run)
            gh = float(np.median([item[3] for item in run]))
            bounds = (x1, y1, x2 - x1, y2 - y1)
            if bounds in seen_rows:
                continue
            seen_rows.add(bounds)
            line_rows.append((run, bounds, gh))
            if len(run) < 5:
                continue
            margin_x = max(2, int((x2 - x1) * 0.13))
            margin_y = max(2, int(gh * 0.40), int(scale_dim * 0.006))
            bx = max(0, x1 - margin_x)
            by = max(0, y1 - margin_y)
            ex = min(image_w, x2 + margin_x)
            ey = min(image_h, y2 + margin_y)
            box = (bx, by, ex - bx, ey - by)
            feature_details: dict[str, object] | None = {} if include_diagnostics else None
            score = _score_box(gray, box, box[2] * box[3], feature_details)
            if score >= 0:
                if feature_details is not None:
                    heights = [int(item[3]) for item in run]
                    gaps = [
                        int(run[index + 1][0] - (run[index][0] + run[index][2]))
                        for index in range(len(run) - 1)
                    ]
                    mean_height = float(np.mean(heights))
                    feature_details.update(
                        {
                            "glyph_component_count": len(run),
                            "row_count": 1,
                            "row_glyph_counts": [len(run)],
                            "glyph_height_mean_px": mean_height,
                            "glyph_height_cv": float(np.std(heights) / mean_height)
                            if mean_height
                            else None,
                            "inter_glyph_gaps_px": gaps,
                            "gap_to_median_height": [float(gap / max(gh, 1.0)) for gap in gaps],
                        }
                    )
                proposals.append(
                    Candidate(
                        box,
                        score,
                        "aligned-glyphs",
                        diagnostic_features=feature_details,
                        raw_box=bounds if include_diagnostics else None,
                    )
                )

    # Pair aligned text rows into a single proposal for two-line registrations.
    for i, (run_a, box_a, char_h_a) in enumerate(line_rows):
        ax, ay, aw, ah = box_a
        for run_b, box_b, char_h_b in line_rows[i + 1 :]:
            bx, by, bw, bh = box_b
            center_gap = abs((by + bh / 2) - (ay + ah / 2))
            if center_gap < 0.65 * max(char_h_a, char_h_b) or center_gap > 3.0 * max(char_h_a, char_h_b):
                continue
            overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            overlap_ratio = overlap / max(1, min(aw, bw))
            if overlap_ratio < 0.28:
                continue
            x1, y1 = max(0, min(ax, bx) - 3), max(0, min(ay, by) - 2)
            x2 = min(image_w, max(ax + aw, bx + bw) + 3)
            y2 = min(image_h, max(ay + ah, by + bh) + 2)
            box = (x1, y1, x2 - x1, y2 - y1)
            feature_details: dict[str, object] | None = {} if include_diagnostics else None
            score = _score_box(gray, box, box[2] * box[3], feature_details)
            if score >= 0:
                if feature_details is not None:
                    rows = (run_a, run_b)
                    row_heights = [
                        [int(item[3]) for item in row]
                        for row in rows
                    ]
                    row_gaps = [
                        [
                            int(row[index + 1][0] - (row[index][0] + row[index][2]))
                            for index in range(len(row) - 1)
                        ]
                        for row in rows
                    ]
                    feature_details.update(
                        {
                            "glyph_component_count": len(run_a) + len(run_b),
                            "row_count": 2,
                            "row_glyph_counts": [len(run_a), len(run_b)],
                            "row_glyph_height_cv": [
                                float(np.std(heights) / np.mean(heights))
                                if np.mean(heights)
                                else None
                                for heights in row_heights
                            ],
                            "row_inter_glyph_gaps_px": row_gaps,
                            "row_gap_to_median_height": [
                                [float(gap / max(height, 1.0)) for gap in gaps]
                                for gaps, height in zip(row_gaps, (char_h_a, char_h_b))
                            ],
                            "row_center_gap_px": float(center_gap),
                            "row_center_gap_to_glyph_height": float(
                                center_gap / max(char_h_a, char_h_b, 1.0)
                            ),
                            "row_horizontal_overlap_ratio": float(overlap_ratio),
                        }
                    )
                proposals.append(
                    Candidate(
                        box,
                        score,
                        "paired-glyph-rows",
                        diagnostic_features=feature_details,
                        raw_box=(
                            min(ax, bx),
                            min(ay, by),
                            max(ax + aw, bx + bw) - min(ax, bx),
                            max(ay + ah, by + bh) - min(ay, by),
                        )
                        if include_diagnostics
                        else None,
                    )
                )
    return proposals


def localize_plates(
    image: np.ndarray,
    limit: int = 30,
    include_diagnostics: bool = True,
) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    """Return scored plate proposals and diagnostic DSP images.

    The detector combines horizontal gradients, Canny edges, and local
    thresholding. No fixed image coordinates or trained plate detector are
    used.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    filtered = cv2.bilateralFilter(clahe, 7, 55, 55)

    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, grad_mask = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, int(width * 0.045)) | 1, max(3, int(height * 0.012)) | 1)
    )
    gradient_map = cv2.morphologyEx(grad_mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    gradient_map = cv2.morphologyEx(
        gradient_map, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    edges = cv2.Canny(filtered, 55, 165)
    edge_map = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    edge_map = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    adaptive = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    adaptive_map = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

    diagnostics = (
        {
            "gray_clahe": clahe,
            "gradient_x": grad_x,
            "gradient_candidates": gradient_map,
            "edges": edges,
            "edge_candidates": edge_map,
            "adaptive_threshold": adaptive,
        }
        if include_diagnostics
        else {}
    )
    masks: Iterable[tuple[str, np.ndarray]] = (
        ("horizontal-gradient", gradient_map),
        ("canny-edges", edge_map),
        ("adaptive-threshold", adaptive_map),
    )

    raw: list[Candidate] = []
    for source, mask in masks:
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 1 or h < 1:
                continue
            contour_area = abs(cv2.contourArea(contour))
            feature_details: dict[str, object] | None = {} if include_diagnostics else None
            score = _score_box(gray, (x, y, w, h), contour_area, feature_details)
            if score >= 0:
                raw.append(
                    Candidate(
                        (x, y, w, h),
                        score,
                        source,
                        diagnostic_features=feature_details,
                    )
                )

    raw.extend(_group_glyph_candidates(gray, include_diagnostics))

    raw.sort(key=lambda item: item.score, reverse=True)
    selected: list[Candidate] = []
    for candidate in raw:
        if all(_iou(candidate.box, kept.box) < 0.68 for kept in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break

    # If contours are sparse on a low-quality image, add proposals around
    # locally bright connected areas. This remains image-derived rather than
    # relying on known plate locations.
    if not selected:
        _, bright = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            box = cv2.boundingRect(contour)
            feature_details: dict[str, object] | None = {} if include_diagnostics else None
            score = _score_box(gray, box, cv2.contourArea(contour), feature_details)
            if score >= 0:
                selected.append(
                    Candidate(
                        box,
                        score,
                        "bright-region-fallback",
                        diagnostic_features=feature_details,
                    )
                )
        selected.sort(key=lambda item: item.score, reverse=True)
        selected = selected[:limit]

    return selected, diagnostics


def crop_candidate(image: np.ndarray, candidate: Candidate) -> np.ndarray:
    """Crop a proposal with a small context margin for OCR robustness."""
    h, w = image.shape[:2]
    x, y, cw, ch = candidate.box
    if candidate.source == "aligned-glyphs":
        # This proposal already encloses the glyph run tightly. Extra grille
        # and plate-frame pixels can hurt OCR on very small source photos.
        px, py = 0, 0
    elif candidate.source == "ocr-text":
        # OCR boxes cover glyphs, while the plate body has a small border and
        # whitespace around that text.
        px, py = max(2, int(cw * 0.16)), max(3, int(ch * 0.35))
    else:
        px, py = max(1, int(cw * 0.035)), max(1, int(ch * 0.10))
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(w, x + cw + px), min(h, y + ch + py)
    return image[y1:y2, x1:x2].copy()
