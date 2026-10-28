"""
Configuration module for the SubClean Video Inpainting Pipeline.
Defines parameters for video ingestion, OCR detection, ROI cropping,
generative inpainting, grain synthesis, and FFmpeg reconstruction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional
import os
import torch


def get_default_device() -> str:
    """Auto-detect the most optimal available computation device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PipelineConfig:
    # ---------------------------------------------------------
    # 1. Video Ingestion & Output Settings
    # ---------------------------------------------------------
    input_video_path: str = ""
    output_video_path: str = "output_cleaned.mp4"
    audio_path_temp: Optional[str] = None
    crf: int = 18                        # Constant Rate Factor for maximum sharpness
    ffmpeg_preset: str = "slow"          # Encoding preset (slow, medium, fast)
    video_codec: str = "libx264"         # H.264 encoder
    audio_codec: str = "copy"            # Lossless copy of original audio track
    pixel_format: str = "yuv420p"        # Universal web/media compatibility

    # ---------------------------------------------------------
    # 2. Text Detection & Mask Generation Settings
    # ---------------------------------------------------------
    # Normalized ROI: (y_min, y_max, x_min, x_max) in range [0.0, 1.0]
    # Default: bottom 28% of the frame, horizontal center 90%
    roi_normalized: Tuple[float, float, float, float] = (0.72, 0.98, 0.05, 0.95)
    roi_padding_px: int = 32             # Extra padding around subtitle box (aligned to 16px)
    
    ocr_engine: str = "easyocr"          # "easyocr" or "paddleocr"
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_confidence_thresh: float = 0.30  # OCR detection confidence threshold
    
    mask_dilation_kernel: int = 11       # Dilation kernel size (px) to cover drop shadows & edges
    mask_feather_sigma: float = 2.5      # Sub-pixel Gaussian feather for seamless alpha blending
    temporal_mask_stabilization: bool = True # Temporal max-pooling across adjacent frames
    temporal_window_size: int = 5        # Number of frames for temporal mask stabilization

    # Multi-Modal Subtitle & Graphic Detection
    enable_emojis: bool = True           # Detect color emojis / icons in caption zones
    enable_stylized_fonts: bool = True   # Detect thick-stroked / stylized social media fonts

    # ---------------------------------------------------------
    # 3. Generative Inpainting Engine Settings
    # ---------------------------------------------------------
    # Options: "propainter", "lama_raft", "lama"
    inpainting_engine: str = "lama_raft"
    
    # Chunking / Sliding Window (OOM prevention on 8-16GB VRAM GPUs)
    chunk_size: int = 15                 # Frames processed per batch
    neighbor_stride: int = 3             # Temporal neighbor stride for optical flow matching
    
    # Anti-Blur Optimization: crop only the subtitle ROI bounding box
    enable_roi_crop_compositing: bool = True
    
    # Color & Texture Harmonization
    enable_harmonization: bool = True    # Harmonizes boundary colors to eliminate patch residuals
    enable_grain_matching: bool = True
    grain_strength: float = 0.35         # Scale factor for matched sensor/compression noise

    # ---------------------------------------------------------
    # 4. Hardware & Memory Management
    # ---------------------------------------------------------
    device: str = field(default_factory=get_default_device)
    fp16: bool = True                    # Mixed-precision FP16 inference for speed & VRAM savings
    max_vram_usage_gb: float = 12.0      # Target maximum VRAM usage before forced garbage collection
    empty_cache_freq: int = 1            # Free PyTorch CUDA cache every N chunks

    # ---------------------------------------------------------
    # 5. Paths & Pretrained Weights
    # ---------------------------------------------------------
    weights_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights")
    lama_weights_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights" / "big-lama.pt")
    propainter_weights_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights" / "ProPainter.pth")
    raft_weights_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "weights" / "raft-things.pth")

    def __post_init__(self):
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        # Ensure device fallback if CUDA is requested but unavailable
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.fp16 = False
        elif self.device == "cpu":
            self.fp16 = False
