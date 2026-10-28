"""
SubClean: Web User Interface (Gradio).
Interactive GUI for removing hard-coded subtitles from videos with zero-blur deep inpainting.
Includes live ROI bounding-box visualization, side-by-side frame preview, and VRAM monitoring.
"""

from pathlib import Path
import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import gradio as gr
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.inpainting.pipeline import VideoInpaintingPipeline
from src.utils.vram_monitor import VRAMMonitor
from src.ffmpeg_utils.audio_extractor import inspect_media_metadata


# Global pipeline instance cache
_GLOBAL_PIPELINE = None
_GLOBAL_CONFIG = None


def get_pipeline(config: PipelineConfig) -> VideoInpaintingPipeline:
    global _GLOBAL_PIPELINE, _GLOBAL_CONFIG
    if _GLOBAL_PIPELINE is None or _GLOBAL_CONFIG != config:
        _GLOBAL_CONFIG = config
        _GLOBAL_PIPELINE = VideoInpaintingPipeline(config)
    return _GLOBAL_PIPELINE


def get_system_banner() -> str:
    """Returns markdown header showing current GPU and hardware status."""
    vram_mon = VRAMMonitor()
    stats = vram_mon.get_memory_stats()
    device_name = stats.get("gpu_name", "CPU")
    vram_str = f"{stats.get('vram_reserved_gb', 0.0):.1f} / {stats.get('vram_total_gb', 0.0):.1f} GB ({stats.get('vram_percent', 0)}%)"
    ram_str = f"{stats.get('ram_used_gb', 0.0):.1f} / {stats.get('ram_total_gb', 0.0):.1f} GB"

    return f"""
    <div style="background-color: #1a1e24; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px; border-left: 4px solid #3b82f6;">
        <span style="font-weight: 600; color: #93c5fd;">Compute Device:</span> <code style="color: #60a5fa;">{device_name}</code> &nbsp;|&nbsp;
        <span style="font-weight: 600; color: #93c5fd;">VRAM Usage:</span> <span style="color: #e2e8f0;">{vram_str}</span> &nbsp;|&nbsp;
        <span style="font-weight: 600; color: #93c5fd;">System RAM:</span> <span style="color: #e2e8f0;">{ram_str}</span> &nbsp;|&nbsp;
        <span style="font-weight: 600; color: #34d399;">Pipeline:</span> <span style="color: #6ee7b7;">FP16 Mixed Precision Active</span>
    </div>
    """


def on_video_uploaded(video_file):
    """Parses video metadata and initializes frame slider range."""
    if not video_file:
        return (
            "No video uploaded",
            gr.update(maximum=0, value=0),
            None,
        )

    video_path = video_file if isinstance(video_file, str) else video_file.name
    meta = inspect_media_metadata(video_path)

    info_md = f"""
    ### 📹 Video Specifications
    - **Resolution:** `{meta.width} × {meta.height}`
    - **Framerate:** `{meta.fps:.2f} FPS`
    - **Frame Count:** `{meta.total_frames}` frames
    - **Duration:** `{meta.duration:.2f}` seconds
    - **Audio Track:** `{"Preserved (Lossless)" if meta.has_audio else "None detected"}`
    """

    # Fetch initial preview frame (frame 0)
    cap = cv2.VideoCapture(video_path)
    ret, frame_bgr = cap.read()
    cap.release()

    preview_img = None
    if ret and frame_bgr is not None:
        preview_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    max_frames = max(0, meta.total_frames - 1)
    return info_md, gr.update(maximum=max_frames, value=min(15, max_frames)), preview_img


def update_roi_preview(video_file, frame_idx, roi_top, roi_bottom, roi_left, roi_right):
    """Draws current ROI bounding box on selected video frame."""
    if not video_file:
        return None

    video_path = video_file if isinstance(video_file, str) else video_file.name
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()

    # Calculate ROI pixel coordinates
    y1 = int(roi_top * h)
    y2 = int(roi_bottom * h)
    x1 = int(roi_left * w)
    x2 = int(roi_right * w)

    # Draw semi-transparent overlay outside ROI
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # Draw bright bounding box border
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Subtitle ROI ({x2 - x1}x{y2 - y1})"
    cv2.putText(vis, label, (x1 + 6, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def test_frame_inpaint(
    video_file,
    frame_idx,
    roi_top,
    roi_bottom,
    roi_left,
    roi_right,
    dilation_px,
    confidence,
    grain_str,
    engine_choice,
    enable_emojis=True,
    enable_stylized_fonts=True,
):
    """Runs instant single-frame detection and inpainting preview."""
    if not video_file:
        return None, None, None, "Please upload a video first."

    video_path = video_file if isinstance(video_file, str) else video_file.name
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        return None, None, None, f"Could not read frame {frame_idx}"

    config = PipelineConfig(
        roi_normalized=(roi_top, roi_bottom, roi_left, roi_right),
        mask_dilation_kernel=int(dilation_px),
        ocr_confidence_thresh=float(confidence),
        enable_grain_matching=True,
        grain_strength=float(grain_str),
        inpainting_engine=engine_choice,
        enable_emojis=bool(enable_emojis),
        enable_stylized_fonts=bool(enable_stylized_fonts),
    )
    pipeline = get_pipeline(config)

    res = pipeline.preview_single_frame(
        frame_bgr=frame_bgr,
        roi_normalized=(roi_top, roi_bottom, roi_left, roi_right),
        dilation_override=int(dilation_px),
        confidence_override=float(confidence),
    )

    before_rgb = cv2.cvtColor(res["roi_preview"], cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2RGB)
    after_rgb = cv2.cvtColor(res["inpainted"], cv2.COLOR_BGR2RGB)

    status_msg = f"✓ Detected {res['detected_count']} subtitle text line(s). Inpainted with {engine_choice}."
    return before_rgb, mask_rgb, after_rgb, status_msg


def process_full_video(
    video_file,
    roi_top,
    roi_bottom,
    roi_left,
    roi_right,
    dilation_px,
    confidence,
    grain_str,
    chunk_size,
    crf_val,
    engine_choice,
    enable_emojis=True,
    enable_stylized_fonts=True,
    progress=gr.Progress(track_tqdm=True),
):
    """Executes the full 5-stage pipeline on the entire uploaded video."""
    if not video_file:
        return None, "Error: No video uploaded."

    try:
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        out_dir = PROJECT_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem
        output_path = str(out_dir / f"cleaned_{stem}.mp4")

        config = PipelineConfig(
            input_video_path=input_path,
            output_video_path=output_path,
            roi_normalized=(roi_top, roi_bottom, roi_left, roi_right),
            mask_dilation_kernel=int(dilation_px),
            ocr_confidence_thresh=float(confidence),
            chunk_size=int(chunk_size),
            crf=int(crf_val),
            enable_grain_matching=True,
            grain_strength=float(grain_str),
            inpainting_engine=engine_choice,
            enable_emojis=bool(enable_emojis),
            enable_stylized_fonts=bool(enable_stylized_fonts),
        )
        pipeline = get_pipeline(config)

        last_status = "Processing initialized..."
        for status in pipeline.process_video(input_path, output_path):
            progress(
                status["progress"] / 100.0,
                desc=f"{status['stage']} | {status['fps']} FPS | ETA: {status['eta_seconds']}s"
            )
            last_status = (
                f"Progress: {status['progress']}% | "
                f"Frame: {status['processed_frames']}/{status['total_frames']} | "
                f"Speed: {status['fps']} FPS | "
                f"Memory: {status['vram_info']}"
            )

        if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
            raise RuntimeError(f"Output video was not generated or is empty: {output_path}")

        return output_path, f"✓ Successfully generated: {Path(output_path).name}\n{last_status}"

    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"\n[ERROR] {e}\n{err_trace}\n")
        raise gr.Error(f"Processing Failed: {str(e)}")


# Build Gradio UI Application
def build_ui():
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: auto !important; }
    .hero-title { font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="SubClean: Zero-Blur Subtitle Removal") as demo:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 8px;">
                <h1 class="hero-title">SubClean: Zero-Blur Video Subtitle Inpainting</h1>
                <p style="color: #94a3b8; font-size: 1.05rem; margin-top: -6px;">
                    Deep-Learning Optical Flow Propagation & Fast Fourier Inpainting for Pristine Background Reconstruction
                </p>
            </div>
            """
        )

        gr.HTML(get_system_banner())

        with gr.Row():
            # Left Column: Upload and Configuration Controls
            with gr.Column(scale=5):
                gr.Markdown("### 1. Video Ingestion")
                video_input = gr.Video(label="Upload Video", sources=["upload"])
                video_info = gr.Markdown("Upload a video to inspect specifications.")

                gr.Markdown("### 2. Region of Interest (ROI) Alignment")
                gr.Markdown("**Quick ROI Presets:**")
                with gr.Row():
                    btn_preset_bottom = gr.Button("📌 Standard Bottom", size="sm")
                    btn_preset_center = gr.Button("📱 TikTok/Reels Center", size="sm", variant="secondary")
                    btn_preset_top = gr.Button("📺 Top Banner", size="sm")
                    btn_preset_full = gr.Button("🌐 Full Frame", size="sm")

                frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Select Frame to Inspect Subtitles")
                
                with gr.Row():
                    roi_top = gr.Slider(0.0, 1.0, value=0.72, step=0.01, label="ROI Top % (y_min)")
                    roi_bottom = gr.Slider(0.0, 1.0, value=0.98, step=0.01, label="ROI Bottom % (y_max)")
                with gr.Row():
                    roi_left = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="ROI Left % (x_min)")
                    roi_right = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="ROI Right % (x_max)")

                gr.Markdown("### 3. Inpainting & Quality Hyperparameters")
                engine_choice = gr.Dropdown(
                    choices=["lama_raft", "propainter", "lama"],
                    value="lama_raft",
                    label="Generative Inpainting Engine",
                    info="lama_raft: Bidirectional optical flow warping + Fast Fourier convolution (Recommended)",
                )

                with gr.Row():
                    enable_emojis_cb = gr.Checkbox(
                        value=True,
                        label="Detect Emojis & Graphic Icons (🔥, 😂, 💀, etc.)",
                        info="Localizes colorful emojis/stickers aligned with captions",
                    )
                    enable_stylized_cb = gr.Checkbox(
                        value=True,
                        label="Detect Stylized / Outline Fonts",
                        info="Captures heavy outline strokes and impact social fonts",
                    )

                with gr.Row():
                    dilation_slider = gr.Slider(1, 35, value=11, step=2, label="Mask Dilation (px)", info="Increase to 11-17px to swallow drop shadows, outlines & glow")
                    conf_slider = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="OCR Confidence Threshold", info="Lower (0.20) catches faint/small text")

                with gr.Row():
                    grain_slider = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Film Grain Matching", info="Synthesizes matched sensor noise into inpainted zone")
                    chunk_slider = gr.Slider(5, 40, value=15, step=5, label="Chunk Batch Size", info="Sliding window size (lower = less VRAM)")

                crf_slider = gr.Slider(14, 26, value=18, step=1, label="FFmpeg CRF Sharpness", info="18 = Near-lossless high-fidelity master quality")

                gr.Markdown(
                    """
                    > 💡 **Anti-Residual Pro Tips:**
                    > - **Vertical TikTok / Reels video?** Click the **📱 TikTok/Reels Center** preset above!
                    > - **Faint dark border/shadow residual?** Increase **Mask Dilation** to **13px – 17px** to fully swallow the drop shadow.
                    > - **Punctuation or faint letters missed?** Lower **Confidence Threshold** to **0.20**.
                    > - **Automatic Enhancements Active**: Outward-only distance feathering, adaptive shadow absorption, and boundary color harmonization are engaged.
                    """
                )

            # Right Column: Live Frame Preview & Full Rendering
            with gr.Column(scale=7):
                gr.Markdown("### 4. Interactive Frame Preview (Before vs. After)")
                with gr.Row():
                    preview_roi = gr.Image(label="Original Frame with Subtitle ROI", interactive=False)
                    preview_mask = gr.Image(label="Dilated Text Mask", interactive=False)
                
                preview_inpainted = gr.Image(label="Inpainted Result (Zero-Blur Composited)", interactive=False)
                
                test_btn = gr.Button("🔬 Test Inpainting on Selected Frame", variant="secondary")
                preview_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### 5. Full Video Reconstruction")
                render_btn = gr.Button("🚀 Process Entire Video (Zero-Blur Master Render)", variant="primary", size="lg")
                
                output_video = gr.Video(label="Final Cleaned Video (Exact Audio Sync)")
                render_status = gr.Markdown("")

        # ---------------- Event Bindings ----------------
        video_input.change(
            fn=on_video_uploaded,
            inputs=[video_input],
            outputs=[video_info, frame_slider, preview_roi],
        )

        # Preset buttons
        btn_preset_bottom.click(
            fn=lambda: (0.72, 0.98, 0.05, 0.95),
            outputs=[roi_top, roi_bottom, roi_left, roi_right],
        )
        btn_preset_center.click(
            fn=lambda: (0.35, 0.75, 0.05, 0.95),
            outputs=[roi_top, roi_bottom, roi_left, roi_right],
        )
        btn_preset_top.click(
            fn=lambda: (0.02, 0.25, 0.05, 0.95),
            outputs=[roi_top, roi_bottom, roi_left, roi_right],
        )
        btn_preset_full.click(
            fn=lambda: (0.05, 0.95, 0.05, 0.95),
            outputs=[roi_top, roi_bottom, roi_left, roi_right],
        )

        # Update ROI overlay when sliders move
        for widget in [frame_slider, roi_top, roi_bottom, roi_left, roi_right]:
            widget.change(
                fn=update_roi_preview,
                inputs=[video_input, frame_slider, roi_top, roi_bottom, roi_left, roi_right],
                outputs=[preview_roi],
            )

        # Test single frame preview
        test_btn.click(
            fn=test_frame_inpaint,
            inputs=[
                video_input, frame_slider, roi_top, roi_bottom, roi_left, roi_right,
                dilation_slider, conf_slider, grain_slider, engine_choice,
                enable_emojis_cb, enable_stylized_cb,
            ],
            outputs=[preview_roi, preview_mask, preview_inpainted, preview_status],
        )

        # Full video render
        render_btn.click(
            fn=process_full_video,
            inputs=[
                video_input, roi_top, roi_bottom, roi_left, roi_right,
                dilation_slider, conf_slider, grain_slider, chunk_slider, crf_slider, engine_choice,
                enable_emojis_cb, enable_stylized_cb,
            ],
            outputs=[output_video, render_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
