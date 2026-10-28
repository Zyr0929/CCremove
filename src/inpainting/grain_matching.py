"""
Film Grain and Sensor Noise Matching Module.
Analyzes surrounding compression artifacts and sensor noise, then synthesizes
statistically matched grain into inpainted patches to eliminate the synthetic "too smooth" look.
"""

from typing import Tuple
import cv2
import numpy as np


class GrainMatcher:
    """
    Measures high-frequency noise statistics from surrounding unmasked video regions
    and injects matched grain into the deep inpainting output.
    """

    def __init__(self, default_strength: float = 0.35):
        self.default_strength = default_strength

    def estimate_noise_sigma(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimates per-channel noise standard deviation from unmasked pixels.
        
        Args:
            image_bgr: Image crop (H, W, 3) uint8.
            mask: Binary mask uint8 (255=masked/subtitle, 0=unmasked background).

        Returns:
            Tuple of standard deviations (sigma_b, sigma_g, sigma_r).
        """
        # Define surrounding context: unmasked pixels
        unmasked = (mask == 0)
        if np.count_nonzero(unmasked) < 100:
            # Not enough unmasked pixels in this crop; return subtle default noise
            return (2.0, 2.0, 2.0)

        # Decompose into low-frequency baseline and high-frequency residuals
        # Using a small Gaussian blur filter
        base = cv2.GaussianBlur(image_bgr, (5, 5), 1.2)
        residuals = image_bgr.astype(np.float32) - base.astype(np.float32)

        # Compute standard deviation for each channel on unmasked pixels only
        sigmas = []
        for c in range(3):
            channel_res = residuals[:, :, c][unmasked]
            # Use median absolute deviation (MAD) or robust standard deviation
            std = float(np.std(channel_res))
            # Clamp to reasonable limits (1.0 to 12.0) to prevent blown-out noise
            std = float(np.clip(std, 1.0, 12.0))
            sigmas.append(std)

        return (sigmas[0], sigmas[1], sigmas[2])

    def apply_matched_grain(
        self,
        original_crop: np.ndarray,
        inpainted_crop: np.ndarray,
        mask: np.ndarray,
        strength: float = None,
    ) -> np.ndarray:
        """
        Synthesizes matched high-frequency grain and adds it to the inpainted region.

        Args:
            original_crop: Pre-inpainting crop (H, W, 3) uint8.
            inpainted_crop: Output from neural inpainter (H, W, 3) uint8.
            mask: Binary mask (H, W) uint8 (255=inpainted area, 0=unmasked).
            strength: Multiplier for grain amplitude (default self.default_strength).

        Returns:
            Inpainted crop (H, W, 3) uint8 with photorealistic matched grain.
        """
        scale = strength if strength is not None else self.default_strength
        if scale <= 0.0:
            return inpainted_crop

        sigma_b, sigma_g, sigma_r = self.estimate_noise_sigma(original_crop, mask)

        # Generate zero-mean Gaussian noise per channel
        h, w = mask.shape[:2]
        noise_b = np.random.normal(0, sigma_b * scale, (h, w))
        noise_g = np.random.normal(0, sigma_g * scale, (h, w))
        noise_r = np.random.normal(0, sigma_r * scale, (h, w))
        noise = np.stack([noise_b, noise_g, noise_r], axis=2)

        # Add noise only to the inpainted pixels
        mask_3d = (mask > 0)[:, :, np.newaxis]
        result = inpainted_crop.astype(np.float32) + (noise * mask_3d)
        result = np.clip(result, 0.0, 255.0).astype(np.uint8)

        return result
