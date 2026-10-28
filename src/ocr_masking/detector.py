"""
Automated Multi-Modal Subtitle, Stylized Font, and Emoji Detection inside Region of Interest (ROI).
Combines deep OCR (EasyOCR / PaddleOCR), color-chrominance emoji localization,
and high-contrast font stroke segmentation to detect all stylized social media fonts and emojis.
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
    polygon: np.ndarray              # Shape (4, 2) in full-frame coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, width, height)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union for two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class SubtitleDetector:
    """
    Multi-Modal Caption & Emoji Detector.
    Detects alphanumeric text, stylized thick-stroked fonts, and color graphical emojis/stickers.
    """

    def __init__(
        self,
        engine: str = "easyocr",
        languages: Optional[List[str]] = None,
        device: str = "cuda",
        confidence_thresh: float = 0.25,
        enable_emojis: bool = True,
        enable_stylized_fonts: bool = True,
    ):
        self.engine = engine.lower()
        self.languages = languages or ["en"]
        self.device = device
        self.confidence_thresh = confidence_thresh
        self.enable_emojis = enable_emojis
        self.enable_stylized_fonts = enable_stylized_fonts
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
        Detects text and emojis inside the frame's ROI using multi-modal detection.
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

        h_crop, w_crop = roi_crop.shape[:2]
        raw_results: List[OCRDetectionResult] = []

        # -----------------------------------------------------------------
        # Stage 1: Deep OCR with Super-Resolution & Sensitive CRAFT Tuning
        # -----------------------------------------------------------------
        if self.engine == "easyocr" and self.reader is not None:
            # Super-resolution upscale for small or mobile video crops (e.g. 360x640, 480x854, 540x960)
            scale = 2.0 if (h_crop < 500 or w_crop < 700) else 1.0
            if scale > 1.0:
                crop_scaled = cv2.resize(roi_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                crop_scaled = roi_crop

            try:
                # Sensitive parameters to capture blocky, outlined, and stylized social fonts
                # Provide native BGR image without contrast clipping to preserve CRAFT character affinity
                detections = self.reader.readtext(
                    crop_scaled,
                    text_threshold=0.15,
                    low_text=0.15,
                    link_threshold=0.15,
                )
                for bbox, text, conf in detections:
                    poly_scaled = np.array(bbox, dtype=np.float32)
                    poly_crop = poly_scaled / scale
                    poly_full = poly_crop + np.array([x_min, y_min], dtype=np.float32)

                    xs = poly_full[:, 0]
                    ys = poly_full[:, 1]
                    bx = int(np.min(xs))
                    by = int(np.min(ys))
                    bw = int(np.max(xs) - bx)
                    bh = int(np.max(ys) - by)

                    # Geometrical sanity checks for subtitle bounding boxes
                    if bw < 14 or bh < 10:
                        continue
                    if bh > int(h_crop * 0.85) or bw > int(w_crop * 0.98):
                        continue
                    aspect = bw / float(bh)
                    if aspect < 0.25 or aspect > 25.0:
                        continue

                    # CRAFT character affinity detection identifies the text region.
                    # If CTC word spelling confidence is lower (due to stylized font styling/symbols),
                    # verify that the patch contains text edge gradient variance rather than flat background.
                    if conf < thresh:
                        lx1 = max(0, bx - x_min)
                        ly1 = max(0, by - y_min)
                        lx2 = min(w_crop, lx1 + bw)
                        ly2 = min(h_crop, ly1 + bh)
                        local_patch = roi_crop[ly1:ly2, lx1:lx2]
                        if local_patch.size > 0:
                            p_gray = cv2.cvtColor(local_patch, cv2.COLOR_BGR2GRAY)
                            lap_var = cv2.Laplacian(p_gray, cv2.CV_64F).var()
                            if lap_var < 35.0:
                                continue

                    raw_results.append(OCRDetectionResult(
                        text=text if text else "<text>",
                        confidence=float(max(conf, 0.78)),
                        polygon=poly_full.astype(np.int32),
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

                        raw_results.append(OCRDetectionResult(
                            text=text,
                            confidence=float(conf),
                            polygon=poly_full,
                            bbox=(bx, by, bw, bh),
                        ))
            except Exception as e:
                logger.error(f"PaddleOCR detection error: {e}")

        # -----------------------------------------------------------------
        # Stage 2: Color Graphic Emoji & Sticker Localization
        # -----------------------------------------------------------------
        if self.enable_emojis:
            emoji_results = self._detect_emojis(roi_crop, x_min, y_min, raw_results)
            raw_results.extend(emoji_results)

        # -----------------------------------------------------------------
        # Stage 3: Visual Caption & Font Stroke Segmenter (Catches fonts OCR misses)
        # -----------------------------------------------------------------
        if self.enable_stylized_fonts:
            stylized_results = self._detect_stylized_captions(roi_crop, x_min, y_min, raw_results)
            raw_results.extend(stylized_results)

        # Deduplicate and merge overlapping candidate boxes
        final_results = self._non_max_suppression(raw_results, iou_thresh=0.35)
        return final_results

    def _detect_emojis(
        self,
        roi_crop: np.ndarray,
        x_offset: int,
        y_offset: int,
        ref_text_results: List[OCRDetectionResult],
    ) -> List[OCRDetectionResult]:
        """
        Detects vivid graphical emojis (faces, fire, money, hearts, icons)
        based on chrominance saturation and spatial co-occurrence with caption text.
        """
        h_crop, w_crop = roi_crop.shape[:2]
        hsv = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Human skin detector to avoid detecting faces/arms as emojis
        ycrcb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        is_skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (sat >= 20) & (sat <= 90)

        # Multi-chrominance emoji color space masks:
        # Yellow faces & hands: high saturation (sat >= 80) clearly separates from skin (sat <= 65)
        is_yellow = (hue >= 14) & (hue <= 42) & (sat >= 80) & (val >= 70) & (~is_skin)
        # Red / Orange flames, hearts, 100 icons
        is_red = ((hue <= 14) | (hue >= 165)) & (sat >= 70) & (val >= 60)
        # Green money, checks, icons
        is_green = (hue >= 38) & (hue <= 85) & (sat >= 50) & (val >= 55)
        # Blue tears, caps, ice
        is_blue = (hue >= 90) & (hue <= 135) & (sat >= 50) & (val >= 60)
        # Purple / Violet
        is_purple = (hue >= 135) & (hue <= 165) & (sat >= 50) & (val >= 60)

        emoji_mask = (is_yellow | is_red | is_green | is_blue | is_purple).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        emoji_mask = cv2.morphologyEx(emoji_mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(emoji_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        emoji_detections = []

        # Find reference font height from existing text detections
        ref_heights = [det.bbox[3] for det in ref_text_results if det.text != "<emoji>"]
        median_font_h = np.median(ref_heights) if ref_heights else (h_crop * 0.25)

        for cnt in contours:
            ex, ey, ew, eh = cv2.boundingRect(cnt)
            aspect = ew / max(1, eh)

            # Emojis are roughly square (0.65 to 1.50) and scaled with font
            if 0.65 <= aspect <= 1.50 and 14 <= eh <= int(h_crop * 0.8):
                # Check height similarity with nearby text or minimum size
                if abs(eh - median_font_h) <= median_font_h * 0.8 or eh >= 18:
                    poly_crop = np.array([[ex, ey], [ex + ew, ey], [ex + ew, ey + eh], [ex, ey + eh]], dtype=np.int32)
                    poly_full = poly_crop + np.array([x_offset, y_offset], dtype=np.int32)
                    emoji_detections.append(OCRDetectionResult(
                        text="<emoji>",
                        confidence=0.88,
                        polygon=poly_full,
                        bbox=(ex + x_offset, ey + y_offset, ew, eh),
                    ))

        return emoji_detections

    def _detect_stylized_captions(
        self,
        roi_crop: np.ndarray,
        x_offset: int,
        y_offset: int,
        existing_results: List[OCRDetectionResult],
    ) -> List[OCRDetectionResult]:
        """
        Visual Morphological Caption Segmenter.
        Captures thick-stroked outline fonts missed by character dictionaries
        while strictly rejecting human skin, tattoos, and continuous clothing.
        """
        h_crop, w_crop = roi_crop.shape[:2]

        # 1. Human skin mask via YCrCb + HSV saturation (excludes arms, hands, faces, tattoos)
        ycrcb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        hsv = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        is_skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (sat >= 20) & (sat <= 95)

        # 2. Text stroke contrast via Top-Hat transform on max color channel
        # Subtitle letters (white, yellow, red, cyan, green) have high contrast against dark outline
        max_rgb = np.max(roi_crop, axis=2)
        k_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        tophat = cv2.morphologyEx(max_rgb, cv2.MORPH_TOPHAT, k_stroke)

        _, stroke_thresh = cv2.threshold(tophat, 55, 255, cv2.THRESH_BINARY)
        # Suppress skin regions
        stroke_thresh[is_skin] = 0

        # Connect individual letters into words horizontally
        k_word = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        connected = cv2.morphologyEx(stroke_thresh, cv2.MORPH_CLOSE, k_word)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stylized_results = []

        for cnt in contours:
            tx, ty, tw, th = cv2.boundingRect(cnt)
            # Minimum word dimensions
            if tw >= 25 and 15 <= th <= int(h_crop * 0.65) and (tw / th) >= 0.6:
                # Patch skin ratio check: if more than 40% of the box is skin, reject
                patch_skin = is_skin[ty:ty+th, tx:tx+tw]
                if np.mean(patch_skin) > 0.40:
                    continue

                # Edge gradient check
                p_gray = cv2.cvtColor(roi_crop[ty:ty+th, tx:tx+tw], cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(p_gray, cv2.CV_64F).var()
                if lap_var < 50.0:
                    continue

                poly_crop = np.array([[tx, ty], [tx + tw, ty], [tx + tw, ty + th], [tx, ty + th]], dtype=np.int32)
                poly_full = poly_crop + np.array([x_offset, y_offset], dtype=np.int32)
                box = (tx + x_offset, ty + y_offset, tw, th)

                # Check if this box is already covered by an existing high-confidence detection
                already_covered = False
                for ex in existing_results:
                    if compute_iou(box, ex.bbox) > 0.3:
                        already_covered = True
                        break

                if not already_covered:
                    stylized_results.append(OCRDetectionResult(
                        text="<stylized_caption>",
                        confidence=0.82,
                        polygon=poly_full,
                        bbox=box,
                    ))

        return stylized_results

    def _non_max_suppression(
        self,
        results: List[OCRDetectionResult],
        iou_thresh: float = 0.35,
    ) -> List[OCRDetectionResult]:
        """Suppresses redundant overlapping bounding boxes, preserving higher confidence detections."""
        if not results:
            return []

        # Sort by confidence descending
        sorted_res = sorted(results, key=lambda r: r.confidence, reverse=True)
        keep = []

        while sorted_res:
            best = sorted_res.pop(0)
            keep.append(best)
            remaining = []
            for r in sorted_res:
                if compute_iou(best.bbox, r.bbox) < iou_thresh:
                    remaining.append(r)
            sorted_res = remaining

        return keep
