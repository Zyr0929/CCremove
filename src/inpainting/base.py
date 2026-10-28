"""
Base Inpainter Interface and High-Resolution ROI Crop & Compositing Engine.
Guarantees zero blur and maximum sharpness by cropping only the subtitle region,
running high-res generative inpainting, and compositing with outward alpha feathering,
matched grain, and boundary color harmonization.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging
import cv2
import numpy as np

from .grain_matching import GrainMatcher

logger = logging.getLogger(__name__)


class BaseInpainter(ABC):
    """Abstract base class for generative inpainting backends."""

    @abstractmethod
    def inpaint_chunk(
        self,
        frames_crop: List[np.ndarray],
        masks_crop: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Inpaints a batch/chunk of image crops and returns reconstructed crops."""
        pass


class ROICropCompositor:
    """
    Anti-Blur & Anti-Residual Optimization Engine.
    Crops ONLY the padded bounding box containing detected subtitles, runs generative inpainting
    at native full resolution, applies boundary color harmonization, injects matched film grain,
    and composites back onto the full frame using strict outward-feathered alpha blending.
    """

    def __init__(
        self,
        padding_px: int = 36,
        feather_sigma: float = 2.5,
        enable_grain: bool = True,
        grain_strength: float = 0.35,
        enable_harmonization: bool = True,
    ):
        self.padding_px = padding_px
        self.feather_sigma = feather_sigma
        self.enable_grain = enable_grain
        self.enable_harmonization = enable_harmonization
        self.grain_matcher = GrainMatcher(default_strength=grain_strength)

    def compute_chunk_crop_bbox(
        self,
        masks: List[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Computes the minimal bounding box enclosing all masks in the chunk,
        expanded by padding and aligned to 16px multiples.
        """
        h_full, w_full = frame_shape[:2]
        
        combined = np.zeros((h_full, w_full), dtype=np.uint8)
        for m in masks:
            if m is not None and np.count_nonzero(m) > 0:
                combined = np.maximum(combined, m)

        y_indices, x_indices = np.where(combined > 0)
        if len(y_indices) == 0:
            return None

        y_min = int(np.min(y_indices)) - self.padding_px
        y_max = int(np.max(y_indices)) + 1 + self.padding_px
        x_min = int(np.min(x_indices)) - self.padding_px
        x_max = int(np.max(x_indices)) + 1 + self.padding_px

        y_min = max(0, y_min)
        y_max = min(h_full, y_max)
        x_min = max(0, x_min)
        x_max = min(w_full, x_max)

        crop_h = y_max - y_min
        crop_w = x_max - x_min

        rem_h = (16 - (crop_h % 16)) % 16
        rem_w = (16 - (crop_w % 16)) % 16

        if rem_h > 0:
            if y_max + rem_h <= h_full:
                y_max += rem_h
            else:
                y_min = max(0, y_min - rem_h)

        if rem_w > 0:
            if x_max + rem_w <= w_full:
                x_max += rem_w
            else:
                x_min = max(0, x_min - rem_w)

        return (y_min, y_max, x_min, x_max)

    def harmonize_boundary_colors(
        self,
        original_crop: np.ndarray,
        inpainted_crop: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Eliminates color and luminance shift between the deep inpainting output
        and the authentic surrounding background (e.g. skin, clothing, lighting gradients).
        """
        if np.count_nonzero(mask) == 0:
            return inpainted_crop

        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        boundary_ring = cv2.dilate(mask, k_outer) - cv2.dilate(mask, k_inner)

        if np.count_nonzero(boundary_ring) < 30:
            return inpainted_crop

        mean_orig = np.mean(original_crop[boundary_ring > 0], axis=0)
        mean_inpaint = np.mean(inpainted_crop[boundary_ring > 0], axis=0)
        color_drift = mean_orig - mean_inpaint

        corrected = inpainted_crop.astype(np.float32) + (color_drift * 0.85)
        return np.clip(corrected, 0.0, 255.0).astype(np.uint8)

    def process_and_composite_chunk(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        inpainter: BaseInpainter,
    ) -> List[np.ndarray]:
        """
        Processes a temporal chunk of full-resolution frames:
        1. Determines tight bounding box for subtitle region.
        2. Crops high-res patch.
        3. Runs generative inpainting on patch.
        4. Applies boundary color harmonization.
        5. Injects matched sensor/film grain.
        6. Composites back onto untouched original frame with strict outward alpha feathering.
        """
        n_frames = len(frames)
        if n_frames == 0:
            return []

        h_full, w_full = frames[0].shape[:2]
        bbox = self.compute_chunk_crop_bbox(masks, (h_full, w_full))

        if bbox is None:
            return [f.copy() for f in frames]

        y_min, y_max, x_min, x_max = bbox

        crops = [f[y_min:y_max, x_min:x_max] for f in frames]
        crop_masks = [m[y_min:y_max, x_min:x_max] for m in masks]

        inpainted_crops = inpainter.inpaint_chunk(crops, crop_masks)

        composited_frames = []
        for orig_frame, orig_crop, inpaint_crop, c_mask in zip(frames, crops, inpainted_crops, crop_masks):
            final_frame = orig_frame.copy()

            if np.count_nonzero(c_mask) == 0:
                composited_frames.append(final_frame)
                continue

            # 1. Boundary Color and Luminance Harmonization (eliminates patch color mismatch)
            if self.enable_harmonization:
                inpaint_crop = self.harmonize_boundary_colors(orig_crop, inpaint_crop, c_mask)

            # 2. Synthesize matched grain onto inpainted crop
            if self.enable_grain:
                inpaint_crop = self.grain_matcher.apply_matched_grain(
                    original_crop=orig_crop,
                    inpainted_crop=inpaint_crop,
                    mask=c_mask,
                )

            # 3. Strict Outward Feathering via Euclidean Distance Transform
            # Inside c_mask: alpha = 1.0 (100% inpainted, zero leak of original shadow/text)
            # Outside c_mask: smooth fade into authentic clean background
            dist = cv2.distanceTransform(255 - c_mask, cv2.DIST_L2, 5)
            feather_px = max(2.0, self.feather_sigma * 2.5)
            alpha = np.clip(1.0 - (dist / feather_px), 0.0, 1.0)
            # Smoothstep curve for natural gradient transition
            alpha = (alpha * alpha * (3.0 - 2.0 * alpha))[:, :, np.newaxis]

            # 4. Alpha blend inpainted patch with untouched background
            blended_crop = (alpha * inpaint_crop.astype(np.float32)) + ((1.0 - alpha) * orig_crop.astype(np.float32))
            blended_crop = np.clip(blended_crop, 0.0, 255.0).astype(np.uint8)

            # 5. Paste back onto full-frame
            final_frame[y_min:y_max, x_min:x_max] = blended_crop
            composited_frames.append(final_frame)

        return composited_frames
