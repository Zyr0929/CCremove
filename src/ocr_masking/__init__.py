"""OCR detection and mask generation module for video subtitles."""
from .detector import SubtitleDetector, OCRDetectionResult
from .mask_generator import MaskGenerator

__all__ = ["SubtitleDetector", "OCRDetectionResult", "MaskGenerator"]
