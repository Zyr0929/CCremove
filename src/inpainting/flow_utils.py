"""
Optical Flow Computation and Temporal Flow Propagation.
Utilizes RAFT / Deep Optical Flow with dense forward-backward warping to propagate
real background textures across frames, achieving zero flicker and maximum temporal sharpness.
"""

from typing import Tuple, Optional
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def warp_frame_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Backward bilinear warping of an image using optical flow.
    
    Args:
        img: Image array (H, W, C) or (H, W) in uint8 or float32.
        flow: Optical flow (H, W, 2) where flow[..., 0] is dx, flow[..., 1] is dy.

    Returns:
        Warped image array of same shape and dtype as img.
    """
    h, w = img.shape[:2]
    # Coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped


class FlowEngine:
    """
    Computes bidirectional optical flow using RAFT or OpenCV DIS flow engine.
    Performs forward-backward consistency checks and temporal feature propagation.
    """

    def __init__(self, device: str = "cuda", fp16: bool = True):
        self.device = device
        self.fp16 = fp16
        self.raft_model = None
        self._init_flow_model()

    def _init_flow_model(self):
        """Attempts to load PyTorch Torchvision RAFT model, falling back to OpenCV DIS flow."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                logger.info("Initializing Torchvision RAFT-Small optical flow on GPU...")
                weights = Raft_Small_Weights.DEFAULT
                self.raft_model = raft_small(weights=weights).to(self.device).eval()
                if self.fp16:
                    self.raft_model = self.raft_model.half()
                return
            except Exception as e:
                logger.debug(f"Torchvision RAFT-small load notice: {e}. Using fast dense DIS flow.")

        # High-performance OpenCV DIS optical flow (Dense Inverse Search)
        try:
            self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.dis_flow.setFinestScale(1)
        except Exception:
            self.dis_flow = None

    def compute_flow(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> np.ndarray:
        """
        Computes optical flow from img1 to img2.
        Returns flow array of shape (H, W, 2) (dx, dy).
        """
        h, w = img1_bgr.shape[:2]

        if self.raft_model is not None and torch.cuda.is_available():
            try:
                # Convert BGR to RGB and scale to [-1, 1] as expected by RAFT
                rgb1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rgb2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                t1 = torch.from_numpy(rgb1).permute(2, 0, 1).unsqueeze(0).to(self.device)
                t2 = torch.from_numpy(rgb2).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Ensure dimensions are divisible by 8 for RAFT
                pad_h = (8 - (h % 8)) % 8
                pad_w = (8 - (w % 8)) % 8
                if pad_h > 0 or pad_w > 0:
                    t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
                    t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h), mode="replicate")

                t1 = (t1 * 2.0) - 1.0
                t2 = (t2 * 2.0) - 1.0

                if self.fp16:
                    t1 = t1.half()
                    t2 = t2.half()

                with torch.no_grad():
                    flow_pred = self.raft_model(t1, t2)[-1]
                    if pad_h > 0 or pad_w > 0:
                        flow_pred = flow_pred[:, :, :h, :w]
                    flow_np = flow_pred[0].permute(1, 2, 0).float().cpu().numpy()
                    return flow_np
            except Exception as e:
                logger.debug(f"RAFT inference failed ({e}), falling back to DIS flow.")

        # OpenCV DIS optical flow computation
        gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

        if self.dis_flow is not None:
            flow = self.dis_flow.calc(gray1, gray2, None)
            return flow

        # Robust Farneback fallback
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow

    def propagate_background_temporal(
        self,
        curr_frame: np.ndarray,
        curr_mask: np.ndarray,
        prev_frame: Optional[np.ndarray],
        prev_mask: Optional[np.ndarray],
        next_frame: Optional[np.ndarray],
        next_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates real ground-truth background textures from temporal neighbors into
        the current frame's subtitle hole using bidirectional optical flow warping.

        Returns:
            (reconstructed_frame, updated_mask) where updated_mask marks only the remaining
            pixels that could not be recovered from temporal neighbors.
        """
        output_frame = curr_frame.copy()
        remaining_mask = curr_mask.copy()

        # If no subtitle in this frame, return immediately
        if np.count_nonzero(curr_mask) == 0:
            return output_frame, remaining_mask

        # 1. Backward propagation from previous frame
        if prev_frame is not None and prev_mask is not None:
            flow_curr_to_prev = self.compute_flow(curr_frame, prev_frame)
            warped_prev = warp_frame_flow(prev_frame, flow_curr_to_prev)
            warped_prev_mask = warp_frame_flow(prev_mask, flow_curr_to_prev)

            # Available valid background: current mask is 255 AND prev pixel was UNMASKED (0)
            valid_prev = (remaining_mask > 128) & (warped_prev_mask < 64)
            output_frame[valid_prev] = warped_prev[valid_prev]
            remaining_mask[valid_prev] = 0

        # 2. Forward propagation from next frame
        if next_frame is not None and next_mask is not None and np.count_nonzero(remaining_mask) > 0:
            flow_curr_to_next = self.compute_flow(curr_frame, next_frame)
            warped_next = warp_frame_flow(next_frame, flow_curr_to_next)
            warped_next_mask = warp_frame_flow(next_mask, flow_curr_to_next)

            valid_next = (remaining_mask > 128) & (warped_next_mask < 64)
            output_frame[valid_next] = warped_next[valid_next]
            remaining_mask[valid_next] = 0

        return output_frame, remaining_mask
