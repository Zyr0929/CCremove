"""
Mask Generation, Morphological Dilation, Shadow Absorption, and Temporal Stabilization.
Produces tight, drop-shadow-inclusive text masks with strict outward-feathered alpha blending.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from .detector import OCRDetectionResult


class MaskGenerator:
    """
    Generates binary and feathered alpha masks from OCR detections.
    Includes adaptive drop-shadow absorption, morphological dilation, and temporal stabilization.
    """

    def __init__(
        self,
        dilation_ksize: int = 11,
        feather_sigma: float = 2.0,
        temporal_window: int = 5,
    ):
        self.dilation_ksize = dilation_ksize
        self.feather_sigma = feather_sigma
        self.temporal_window = temporal_window

    def generate_single_mask(
        self,
        frame_shape: Tuple[int, int],
        detections: List[OCRDetectionResult],
        frame_bgr: Optional[np.ndarray] = None,
        dilation_override: Optional[int] = None,
    ) -> np.ndarray:
        """
        Creates a binary uint8 mask (0=background, 255=subtitle text + shadows) for a single frame.
        
        Args:
            frame_shape: (height, width) of the full frame.
            detections: List of OCRDetectionResult with polygons in full frame space.
            frame_bgr: Optional full frame to run adaptive shadow/halo absorption.
            dilation_override: Optional custom dilation kernel size.

        Returns:
            Binary mask uint8 of shape (height, width).
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not detections:
            return mask

        # 1. Draw filled polygons for all detected text boxes
        for det in detections:
            pts = det.polygon.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

        # 2. Adaptive drop-shadow & outline stroke absorption
        if frame_bgr is not None:
            mask = self._absorb_text_shadows_and_outlines(frame_bgr, mask)

        # 3. Morphological dilation to guarantee complete coverage of anti-aliasing & glow
        ksize = dilation_override if dilation_override is not None else self.dilation_ksize
        if ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _absorb_text_shadows_and_outlines(self, frame_bgr: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """
        Inspects the perimeter of the detected text for dark drop shadows or heavy outlines
        and expands the mask to fully absorb them.
        """
        if np.count_nonzero(base_mask) == 0:
            return base_mask

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Analyze border band (8px outward)
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        band = cv2.dilate(base_mask, k_outer) - cv2.dilate(base_mask, k_inner)

        if np.count_nonzero(band) < 50:
            return base_mask

        # Estimate average background brightness in surrounding area
        bg_mean = np.mean(gray[band > 0])
        # Dark drop shadows typically have luminance lower than 0.7 * bg_mean
        shadow_thresh = max(20, int(bg_mean * 0.75))

        # Check immediate shadow expansion zone (down and right, typical subtitle shadow angle)
        k_shadow = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        shadow_zone = cv2.dilate(base_mask, k_shadow) - base_mask

        is_shadow = (gray < shadow_thresh) & (shadow_zone > 0)
        expanded = base_mask.copy()
        expanded[is_shadow] = 255

        return expanded

    def stabilize_masks_temporal(
        self,
        masks: List[np.ndarray],
        window_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Performs temporal max-pooling across a sliding window of frames.
        Prevents subtitle mask flickering and drops caused by per-frame OCR variance.
        """
        n_frames = len(masks)
        if n_frames <= 1:
            return masks

        w_radius = (window_size if window_size is not None else self.temporal_window) // 2
        if w_radius < 1:
            return masks

        stacked = np.stack(masks, axis=0)  # Shape (T, H, W)
        stabilized = []

        for t in range(n_frames):
            t_min = max(0, t - w_radius)
            t_max = min(n_frames, t + w_radius + 1)
            local_max = np.max(stacked[t_min:t_max], axis=0)
            stabilized.append(local_max)

        return stabilized

    def compute_outward_feathered_alpha(
        self,
        binary_mask: np.ndarray,
        feather_width_px: float = 6.0,
    ) -> np.ndarray:
        """
        Computes strict OUTWARD-only feathered alpha mask.
        Inside binary_mask: alpha is exactly 1.0 (100% inpainted, zero original shadow leakage).
        Outside binary_mask: alpha smoothly decays to 0.0 into the authentic clean background.
        """
        if np.count_nonzero(binary_mask) == 0:
            return np.zeros(binary_mask.shape[:2], dtype=np.float32)

        # Distance transform on inverted mask (distance from mask boundary outwards into background)
        dist = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)
        
        # Smooth cosine / linear falloff over feather_width_px
        alpha = np.clip(1.0 - (dist / max(1.0, feather_width_px)), 0.0, 1.0)
        # Apply smoothstep for natural organic boundary transition
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return alpha
