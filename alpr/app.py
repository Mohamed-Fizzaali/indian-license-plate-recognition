"""App-facing result adapter for the existing ALPR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import recognize_image


@dataclass
class AppResult:
    plate_number: str
    confidence: float
    valid_indian_format: bool
    bounding_box: tuple[int, int, int, int] | None
    plate_crop: np.ndarray | None
    ocr_method: str
    processing_status: str
    diagnostics: dict[str, np.ndarray] | None = None
    error: str | None = None


def recognize_for_app(
    image: np.ndarray | str | Path,
    include_diagnostics: bool = False,
) -> AppResult:
    """Recognize a plate for an app bridge without saving diagnostic files."""
    try:
        result = recognize_image(image, include_diagnostics=include_diagnostics)
    except Exception as exc:
        return AppResult(
            plate_number="",
            confidence=0.0,
            valid_indian_format=False,
            bounding_box=None,
            plate_crop=None,
            ocr_method="error",
            processing_status="error",
            diagnostics=None,
            error=str(exc),
        )

    if result.plate_box is None:
        status = "plate_not_found"
    elif not result.plate_number:
        status = "recognition_failed"
    else:
        status = "success"

    return AppResult(
        plate_number=result.plate_number,
        confidence=result.confidence,
        valid_indian_format=result.valid_indian_format,
        bounding_box=result.plate_box,
        plate_crop=result.plate_crop,
        ocr_method=result.ocr_method,
        processing_status=status,
        diagnostics=result.diagnostics if include_diagnostics else None,
    )
