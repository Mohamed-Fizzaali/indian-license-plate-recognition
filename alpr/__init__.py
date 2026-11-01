"""Classical image-processing based license plate recognition."""

from .pipeline import ALPRResult, recognize_image
from .app import AppResult, recognize_for_app

__all__ = ["ALPRResult", "AppResult", "recognize_image", "recognize_for_app"]
