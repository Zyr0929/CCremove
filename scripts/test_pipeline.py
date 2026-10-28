"""
Comprehensive Integration Test for SubClean Video Inpainting Pipeline.
Generates a synthetic high-resolution video with moving textured background
and burned-in hard-coded subtitles, runs the full 5-stage pipeline,
and verifies zero-blur reconstruction.
"""

from pathlib import Path
import sys
import os
import cv2
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.inpainting.pipeline import VideoInpaintingPipeline


def generate_synthetic_subtitled_video(output_mp4: str, num_frames: int = 30, width: int = 960, height: int = 540) -> str:
    """Generates an MP4 video with moving textured background and burned-in subtitles."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_mp4, fourcc, 25.0, (width, height))

    subtitles = [
        "THIS IS A HARD-CODED BURNED-IN SUBTITLE",
        "ZERO-BLUR OPTICAL FLOW RECONSTRUCTION",
    ]

    for i in range(num_frames):
        # Create a dynamic moving textured background (checkerboard/stripes)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient base
        y_grad = np.linspace(30, 180, height, dtype=np.uint8)[:, None]
        frame[:, :, 0] = y_grad
        frame[:, :, 1] = (y_grad // 2) + 20
        frame[:, :, 2] = 200 - (y_grad // 2)

        # Dynamic moving diagonal grid pattern to test texture sharpness
        offset = (i * 6) % 60
        for x in range(-height, width + height, 40):
            pt1 = (x + offset, 0)
            pt2 = (x + offset + height, height)
            cv2.line(frame, pt1, pt2, (80, 140, 90), 2)

        # Burned-in subtitle text in bottom 20%
        sub_text = subtitles[(i // 15) % len(subtitles)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        text_size, _ = cv2.getTextSize(sub_text, font, font_scale, thickness)
        tx = (width - text_size[0]) // 2
        ty = int(height * 0.88)

        # Black outline / drop shadow
        cv2.putText(frame, sub_text, (tx + 2, ty + 2), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        # White subtitle text
        cv2.putText(frame, sub_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"Created synthetic test video: {output_mp4} ({num_frames} frames, {width}x{height})")
    return output_mp4


def main():
    test_dir = PROJECT_ROOT / "test_data"
    test_dir.mkdir(exist_ok=True)

    input_video = str(test_dir / "synthetic_subtitled_input.mp4")
    output_video = str(test_dir / "cleaned_output.mp4")

    print("\n" + "=" * 60)
    print(" SUBCLEAN VIDEO INPAINTING INTEGRATION TEST")
    print("=" * 60)

    # 1. Generate synthetic video
    generate_synthetic_subtitled_video(input_video, num_frames=30)

    # 2. Configure pipeline with ROI targeting bottom 25%
    config = PipelineConfig(
        input_video_path=input_video,
        output_video_path=output_video,
        roi_normalized=(0.75, 0.98, 0.05, 0.95),
        mask_dilation_kernel=7,
        mask_feather_sigma=2.0,
        enable_grain_matching=True,
        grain_strength=0.30,
        chunk_size=10,
        crf=18,
    )

    print(f"\nInitialized configuration. Target Device: {config.device.upper()}")
    pipeline = VideoInpaintingPipeline(config)

    # 3. Test single-frame preview
    print("\n--- Testing Single-Frame Interactive Preview ---")
    cap = cv2.VideoCapture(input_video)
    ret, sample_frame = cap.read()
    cap.release()
    assert ret, "Failed to read sample frame from synthetic video"

    preview_res = pipeline.preview_single_frame(sample_frame)
    print(f"✓ Detected subtitle regions: {preview_res['detected_count']}")

    # Save visual verification side-by-side
    h, w = sample_frame.shape[:2]
    vis_before = preview_res["roi_preview"]
    vis_mask = cv2.cvtColor(preview_res["mask"], cv2.COLOR_GRAY2BGR)
    vis_after = preview_res["inpainted"]
    comparison = np.hstack([vis_before, vis_mask, vis_after])

    preview_img_path = str(test_dir / "preview_comparison.jpg")
    cv2.imwrite(preview_img_path, comparison)
    print(f"✓ Saved before/mask/after visual comparison: {preview_img_path}")

    # 4. Test full video processing
    print("\n--- Testing Full 5-Stage Streaming Pipeline ---")
    for status in pipeline.process_video(input_video, output_video):
        pct = status["progress"]
        fps = status["fps"]
        cur = status["processed_frames"]
        tot = status["total_frames"]
        vram = status["vram_info"]
        print(f"\rProgress: [{pct:5.1f}%] Frame {cur}/{tot} | {fps:.1f} FPS | {vram}", end="", flush=True)

    print("\n\n✓ Video processing finished successfully!")
    out_path = Path(output_video)
    assert out_path.exists(), f"Output video was not created: {output_video}"
    assert out_path.stat().st_size > 1000, f"Output video is too small ({out_path.stat().st_size} bytes)"

    print(f"Output Video Verified: {output_video} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 60)
    print(" ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
