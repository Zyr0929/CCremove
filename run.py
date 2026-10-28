"""
Command-Line Interface (CLI) for SubClean.
Production-ready batch and headless processing tool for removing hard-coded subtitles from videos.
"""

import argparse
import sys
import os
import time
from pathlib import Path
import logging

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.config import PipelineConfig
from src.inpainting.pipeline import VideoInpaintingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SubClean-CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SubClean: Deep Generative Video Caption & Subtitle Removal Pipeline (Zero-Blur)."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video (.mp4, .mkv, .mov, etc.)")
    parser.add_argument("--output", "-o", type=str, default="output_cleaned.mp4", help="Path to output video file")
    
    # ROI arguments
    parser.add_argument("--roi-top", type=float, default=0.72, help="ROI Top boundary ratio [0.0 - 1.0] (default: 0.72)")
    parser.add_argument("--roi-bottom", type=float, default=0.98, help="ROI Bottom boundary ratio [0.0 - 1.0] (default: 0.98)")
    parser.add_argument("--roi-left", type=float, default=0.05, help="ROI Left boundary ratio [0.0 - 1.0] (default: 0.05)")
    parser.add_argument("--roi-right", type=float, default=0.95, help="ROI Right boundary ratio [0.0 - 1.0] (default: 0.95)")
    
    # Processing parameters
    parser.add_argument("--engine", type=str, default="lama_raft", choices=["lama_raft", "propainter", "lama"],
                        help="Inpainting engine (default: lama_raft)")
    parser.add_argument("--ocr-engine", type=str, default="easyocr", choices=["easyocr", "paddleocr"],
                        help="OCR engine (default: easyocr)")
    parser.add_argument("--confidence", type=float, default=0.30, help="OCR confidence threshold (default: 0.30)")
    parser.add_argument("--dilation", type=int, default=7, help="Mask dilation kernel size in px (default: 7)")
    parser.add_argument("--chunk-size", type=int, default=15, help="Frame batch chunk size (default: 15)")
    parser.add_argument("--grain-strength", type=float, default=0.35, help="Film grain matching strength (default: 0.35)")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 CRF sharpness factor (default: 18)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")

    return parser.parse_args()


def main():
    args = parse_args()
    input_p = Path(args.input)
    if not input_p.exists():
        logger.error(f"Input video file not found: {args.input}")
        sys.exit(1)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = PipelineConfig(
        input_video_path=str(input_p),
        output_video_path=args.output,
        roi_normalized=(args.roi_top, args.roi_bottom, args.roi_left, args.roi_right),
        ocr_engine=args.ocr_engine,
        ocr_confidence_thresh=args.confidence,
        mask_dilation_kernel=args.dilation,
        inpainting_engine=args.engine,
        chunk_size=args.chunk_size,
        grain_strength=args.grain_strength,
        crf=args.crf,
        device=device,
    )

    logger.info("=" * 65)
    logger.info(" SUBCLEAN: ZERO-BLUR VIDEO CAPTION REMOVAL PIPELINE")
    logger.info(f" Input Video:   {input_p.resolve()}")
    logger.info(f" Output Video:  {Path(args.output).resolve()}")
    logger.info(f" Engine:        {config.inpainting_engine} | Device: {config.device.upper()}")
    logger.info(f" Subtitle ROI:  Top={args.roi_top:.2f}, Bottom={args.roi_bottom:.2f}, "
                f"Left={args.roi_left:.2f}, Right={args.roi_right:.2f}")
    logger.info(f" Dilation:      {config.mask_dilation_kernel}px | CRF: {config.crf} (Maximum Sharpness)")
    logger.info("=" * 65)

    pipeline = VideoInpaintingPipeline(config)

    print("\nStarting video processing stream...\n")
    for status in pipeline.process_video(str(input_p), args.output):
        pct = status["progress"]
        cur = status["processed_frames"]
        tot = status["total_frames"]
        fps = status["fps"]
        eta = status["eta_seconds"]
        vram = status["vram_info"]
        print(f"\rProgress: [{pct:5.1f}%] Frame {cur}/{tot} ({fps:.1f} FPS, ETA: {eta}s) | {vram}", end="", flush=True)

    print(f"\n\nProcessing complete! Video saved to: {Path(args.output).resolve()}\n")


if __name__ == "__main__":
    main()
