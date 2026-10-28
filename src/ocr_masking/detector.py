"""
Automated Subtitle and Text Detection inside Region of Interest (ROI).
Integrates EasyOCR / PaddleOCR with coordinate translation and confidence filtering.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRDetectionResult:
    text: str
    confidence: float
    polygon: np.ndarray       # Shape (4, 2) in full-frame coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, width, height)


class SubtitleDetector:
    """
    Detects subtitle text exclusively within a specified Region of Interest (ROI).
    Extracts tight polygons and maps them back to full-frame coordinates.
    """

    def __init__(
        self,
        engine: str = "easyocr",
        languages: Optional[List[str]] = None,
        device: str = "cuda",
        confidence_thresh: float = 0.30,
    ):
        self.engine = engine.lower()
        self.languages = languages or ["en"]
        self.device = device
        self.confidence_thresh = confidence_thresh
        self.reader = None
        self._init_engine()

    def _init_engine(self):
        """Initializes the underlying OCR engine."""
        use_gpu = (self.device == "cuda")

        if self.engine == "easyocr":
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR with languages={self.languages}, gpu={use_gpu}...")
                self.reader = easyocr.Reader(self.languages, gpu=use_gpu)
            except ImportError:
                logger.warning("EasyOCR is not installed. Will attempt PaddleOCR or lightweight fallback.")
                self.engine = "paddleocr"

        if self.engine == "paddleocr" and self.reader is None:
            try:
                from paddleocr import PaddleOCR
                lang = self.languages[0] if self.languages else "en"
                logger.info(f"Initializing PaddleOCR with lang={lang}, use_gpu={use_gpu}...")
                self.reader = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu, show_log=False)
            except ImportError:
                logger.warning("PaddleOCR is not installed. Falling back to heuristic text edge detector.")
                self.engine = "fallback"

    def detect(
        self,
        frame_bgr: np.ndarray,
        roi_normalized: Tuple[float, float, float, float] = (0.72, 0.98, 0.05, 0.95),
        confidence_override: Optional[float] = None,
    ) -> List[OCRDetectionResult]:
        """
        Detects text inside the frame's ROI.
        
        Args:
            frame_bgr: Full-resolution video frame (H, W, 3) in BGR format.
            roi_normalized: (ymin_ratio, ymax_ratio, xmin_ratio, xmax_ratio) in [0.0, 1.0].
            confidence_override: Optional confidence threshold to override default.

        Returns:
            List of OCRDetectionResult with polygons mapped to full frame space.
        """
        h_full, w_full = frame_bgr.shape[:2]
        thresh = confidence_override if confidence_override is not None else self.confidence_thresh

        # Calculate ROI pixel coordinates
        y_min = int(np.clip(roi_normalized[0] * h_full, 0, h_full - 1))
        y_max = int(np.clip(roi_normalized[1] * h_full, y_min + 1, h_full))
        x_min = int(np.clip(roi_normalized[2] * w_full, 0, w_full - 1))
        x_max = int(np.clip(roi_normalized[3] * w_full, x_min + 1, w_full))

        roi_crop = frame_bgr[y_min:y_max, x_min:x_max]
        if roi_crop.size == 0:
            return []

        results: List[OCRDetectionResult] = []

        if self.engine == "easyocr" and self.reader is not None:
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            try:
                detections = self.reader.readtext(roi_rgb)
                for bbox, text, conf in detections:
                    if conf < thresh:
                        continue
                    # bbox is list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in ROI space
                    poly_crop = np.array(bbox, dtype=np.int32)
                    poly_full = poly_crop + np.array([x_min, y_min], dtype=np.int32)

                    xs = poly_full[:, 0]
                    ys = poly_full[:, 1]
                    bx = int(np.min(xs))
                    by = int(np.min(ys))
                    bw = int(np.max(xs) - bx)
                    bh = int(np.max(ys) - by)

                    results.append(OCRDetectionResult(
                        text=text,
                        confidence=float(conf),
                        polygon=poly_full,
                        bbox=(bx, by, bw, bh),
                    ))
            except Exception as e:
                logger.error(f"EasyOCR detection error: {e}")

        elif self.engine == "paddleocr" and self.reader is not None:
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            try:
                raw_res = self.reader.ocr(roi_rgb, cls=False)
                if raw_res and raw_res[0]:
                    for line in raw_res[0]:
                        box_points = line[0]
                        text, conf = line[1]
                        if conf < thresh:
                            continue
                        poly_crop = np.array(box_points, dtype=np.int32)
                        poly_full = poly_crop + np.array([x_min, y_min], dtype=np.int32)

                        xs = poly_full[:, 0]
                        ys = poly_full[:, 1]
                        bx = int(np.min(xs))
                        by = int(np.min(ys))
                        bw = int(np.max(xs) - bx)
                        bh = int(np.max(ys) - by)

                        results.append(OCRDetectionResult(
                            text=text,
                            confidence=float(conf),
                            polygon=poly_full,
                            bbox=(bx, by, bw, bh),
                        ))
            except Exception as e:
                logger.error(f"PaddleOCR detection error: {e}")

        else:
            # Fallback heuristic detector: MSER + Edge localization for high-contrast subtitles
            results = self._detect_heuristic_fallback(roi_crop, x_min, y_min)

        return results

    def _detect_heuristic_fallback(
        self, roi_crop: np.ndarray, x_offset: int, y_offset: int
    ) -> List[OCRDetectionResult]:
        """
        Lightweight computer vision text localization fallback using edge contrast & morphology.
        Activated only if EasyOCR or PaddleOCR are not installed.
        """
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        # Gradient magnitude / Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, thresh = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
        # Connect text horizontal strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        h_roi, w_roi = roi_crop.shape[:2]
        min_w = int(w_roi * 0.04)
        min_h = int(h_roi * 0.05)
        max_h = int(h_roi * 0.8)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= min_w and min_h <= h <= max_h:
                poly_crop = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                poly_full = poly_crop + np.array([x_offset, y_offset], dtype=np.int32)
                results.append(OCRDetectionResult(
                    text="<detected_subtitle>",
                    confidence=0.85,
                    polygon=poly_full,
                    bbox=(x + x_offset, y + y_offset, w, h),
                ))

        return results
