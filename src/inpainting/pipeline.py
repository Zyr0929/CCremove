"""
End-to-End 5-Stage Video Inpainting Pipeline Orchestrator.
Connects Video Ingestion, Audio Separation, OCR Masking, Generative Inpainting,
Grain Synthesis, and FFmpeg Reconstruction into a streaming workflow with VRAM management.
"""

from pathlib import Path
from typing import Generator, Dict, Any, Optional, Callable
import logging
import os
import time
import cv2
import numpy as np
import torch

from ..config import PipelineConfig
from ..ffmpeg_utils.audio_extractor import inspect_media_metadata, extract_lossless_audio
from ..ffmpeg_utils.video_writer import FFmpegPipeWriter
from ..ocr_masking.detector import SubtitleDetector
from ..ocr_masking.mask_generator import MaskGenerator
from ..utils.video_stream import VideoStreamReader
from ..utils.vram_monitor import VRAMMonitor, clean_vram
from .base import BaseInpainter, ROICropCompositor
from .lama_engine import LaMaInpainter
from .propainter_engine import ProPainterInpainter

logger = logging.getLogger(__name__)


class InpainterWrapper(BaseInpainter):
    """Wraps ProPainter or LaMa engines into the common BaseInpainter interface."""

    def __init__(self, engine_name: str, config: PipelineConfig):
        self.engine_name = engine_name.lower()
        self.config = config

        if "propainter" in self.engine_name:
            self.engine = ProPainterInpainter(
                weights_path=config.propainter_weights_path,
                device=config.device,
                fp16=config.fp16,
                chunk_size=config.chunk_size,
                neighbor_stride=config.neighbor_stride,
            )
        elif "lama" in self.engine_name:
            self.engine = LaMaInpainter(
                weights_path=config.lama_weights_path,
                device=config.device,
                fp16=config.fp16,
            )
        else:
            # Default to ProPainter (or fallback)
            self.engine = ProPainterInpainter(
                device=config.device,
                fp16=config.fp16,
                chunk_size=config.chunk_size,
            )

    def inpaint_chunk(self, frames_crop, masks_crop):
        if hasattr(self.engine, "inpaint_chunk"):
            return self.engine.inpaint_chunk(frames_crop, masks_crop)
        elif hasattr(self.engine, "inpaint_crop"):
            return [self.engine.inpaint_crop(f, m) for f, m in zip(frames_crop, masks_crop)]
        return frames_crop


class VideoInpaintingPipeline:
    """
    Main 5-stage production video inpainting pipeline.
    Streams frames from input to output with zero OOM risk and maximum sharpness.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.vram_monitor = VRAMMonitor(device=self.config.device)
        self.detector = SubtitleDetector(
            engine=self.config.ocr_engine,
            languages=self.config.ocr_languages,
            device=self.config.device,
            confidence_thresh=self.config.ocr_confidence_thresh,
            enable_emojis=self.config.enable_emojis,
            enable_stylized_fonts=self.config.enable_stylized_fonts,
        )
        self.mask_generator = MaskGenerator(
            dilation_ksize=self.config.mask_dilation_kernel,
            feather_sigma=self.config.mask_feather_sigma,
            temporal_window=self.config.temporal_window_size,
        )
        self.compositor = ROICropCompositor(
            padding_px=self.config.roi_padding_px,
            feather_sigma=self.config.mask_feather_sigma,
            enable_grain=self.config.enable_grain_matching,
            grain_strength=self.config.grain_strength,
        )
        self.inpainter = InpainterWrapper(
            engine_name=self.config.inpainting_engine,
            config=self.config,
        )

    def preview_single_frame(
        self,
        frame_bgr: np.ndarray,
        roi_normalized: Optional[tuple] = None,
        dilation_override: Optional[int] = None,
        confidence_override: Optional[float] = None,
        adjacent_frames: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Runs OCR text detection, mask generation, and high-resolution inpainting
        on a single preview frame.
        Optionally uses adjacent_frames for temporal mask stabilization.
        """
        roi = roi_normalized or self.config.roi_normalized
        h, w = frame_bgr.shape[:2]

        # 1. Detect text on target frame
        detections = self.detector.detect(
            frame_bgr=frame_bgr,
            roi_normalized=roi,
            confidence_override=confidence_override,
        )

        # 2. Generate dilated mask with shadow absorption
        mask = self.mask_generator.generate_single_mask(
            frame_shape=(h, w),
            detections=detections,
            frame_bgr=frame_bgr,
            dilation_override=dilation_override,
        )

        # 2b. Temporal mask stabilization if adjacent frames provided
        if adjacent_frames and len(adjacent_frames) > 1 and self.config.temporal_mask_stabilization:
            window_masks = []
            for f in adjacent_frames:
                if f is frame_bgr:
                    window_masks.append(mask)
                else:
                    d = self.detector.detect(f, roi_normalized=roi, confidence_override=confidence_override)
                    m = self.mask_generator.generate_single_mask(
                        frame_shape=(h, w),
                        detections=d,
                        frame_bgr=f,
                        dilation_override=dilation_override,
                    )
                    window_masks.append(m)
            stabilized = self.mask_generator.stabilize_masks_temporal(window_masks, window_size=3)
            # Find index of target frame
            target_idx = adjacent_frames.index(frame_bgr) if frame_bgr in adjacent_frames else (len(adjacent_frames) // 2)
            mask = stabilized[target_idx]

        # 3. Create ROI preview overlay
        roi_vis = frame_bgr.copy()
        y1 = int(roi[0] * h)
        y2 = int(roi[1] * h)
        x1 = int(roi[2] * w)
        x2 = int(roi[3] * w)
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for det in detections:
            cv2.polylines(roi_vis, [det.polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        # 4. Inpaint using ROI Crop Compositor
        inpainted_frames = self.compositor.process_and_composite_chunk(
            frames=[frame_bgr],
            masks=[mask],
            inpainter=self.inpainter,
        )
        inpainted_result = inpainted_frames[0]

        return {
            "original": frame_bgr,
            "roi_preview": roi_vis,
            "mask": mask,
            "inpainted": inpainted_result,
            "detected_count": len(detections),
        }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Generator[Dict[str, Any], None, str]:
        """
        Executes the complete 5-stage pipeline on the input video.
        Streams frames chunk-by-chunk to prevent high RAM/VRAM usage.
        
        Yields status dicts with progress percentage, current frame, FPS, ETA, and VRAM stats.
        Returns final output video path.
        """
        input_path = str(input_path)
        output_path = str(output_path)
        start_time = time.time()

        # Step 1: Video Ingestion & Metadata Inspection
        logger.info(f"Stage 1: Ingesting video {input_path}...")
        metadata = inspect_media_metadata(input_path)
        total_frames = metadata.total_frames
        if total_frames <= 0:
            # Fallback estimation if total frames unknown
            total_frames = 1000

        # Step 2: Lossless Audio Track Extraction
        temp_dir = Path(output_path).parent / ".temp_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_audio_file = temp_dir / f"extracted_{Path(input_path).stem}.aac"
        has_audio = False

        if metadata.has_audio:
            logger.info("Stage 1b: Extracting original lossless audio track...")
            has_audio = extract_lossless_audio(input_path, str(temp_audio_file))

        audio_to_remux = str(temp_audio_file) if has_audio else None

        # Step 3: Initialize Video Reader & FFmpeg Pipe Writer
        logger.info(f"Stage 4: Initializing FFmpeg Pipe Writer (CRF {self.config.crf}, {metadata.fps:.2f} FPS)...")
        writer = FFmpegPipeWriter(
            output_path=output_path,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            audio_path=audio_to_remux,
            crf=self.config.crf,
            preset=self.config.ffmpeg_preset,
            video_codec=self.config.video_codec,
            pixel_format=self.config.pixel_format,
        )

        reader = VideoStreamReader(input_path)
        chunk_size = self.config.chunk_size
        processed_frames = 0

        try:
            for chunk_indices, chunk_frames in reader.read_chunks(chunk_size=chunk_size):
                chunk_start_t = time.time()
                n_chunk = len(chunk_frames)

                # Stage 2: OCR Text Detection for each frame in chunk
                chunk_masks = []
                for f in chunk_frames:
                    dets = self.detector.detect(
                        frame_bgr=f,
                        roi_normalized=self.config.roi_normalized,
                    )
                    mask = self.mask_generator.generate_single_mask(
                        frame_shape=(metadata.height, metadata.width),
                        detections=dets,
                        frame_bgr=f,
                    )
                    chunk_masks.append(mask)

                # Temporal Mask Stabilization (smooth out character jitter)
                if self.config.temporal_mask_stabilization and len(chunk_masks) > 1:
                    chunk_masks = self.mask_generator.stabilize_masks_temporal(
                        chunk_masks,
                        window_size=self.config.temporal_window_size,
                    )

                # Stage 3: Generative Inpainting with ROI Crop & Compositing
                inpainted_chunk = self.compositor.process_and_composite_chunk(
                    frames=chunk_frames,
                    masks=chunk_masks,
                    inpainter=self.inpainter,
                )

                # Stage 4: Stream frames directly into FFmpeg Pipe Writer
                for frame_out in inpainted_chunk:
                    writer.write_frame(frame_out)
                    processed_frames += 1

                # Progress & Speed Metrics
                elapsed = time.time() - start_time
                fps_processing = processed_frames / elapsed if elapsed > 0 else 0.0
                remaining_frames = max(0, total_frames - processed_frames)
                eta_seconds = remaining_frames / fps_processing if fps_processing > 0 else 0.0
                progress_pct = min(100.0, round((processed_frames / total_frames) * 100.0, 1))

                mem_stats = self.vram_monitor.get_memory_stats()
                vram_info = self.vram_monitor.format_summary()

                status = {
                    "progress": progress_pct,
                    "processed_frames": processed_frames,
                    "total_frames": total_frames,
                    "fps": round(fps_processing, 1),
                    "eta_seconds": int(eta_seconds),
                    "vram_info": vram_info,
                    "mem_stats": mem_stats,
                    "stage": f"Inpainting chunk ({processed_frames}/{total_frames})",
                    "preview_frame": inpainted_chunk[-1] if inpainted_chunk else None,
                }

                if progress_callback:
                    progress_callback(status)
                yield status

                # Memory Clean-up
                if self.config.empty_cache_freq > 0:
                    clean_vram()

        finally:
            try:
                reader.close()
            except Exception:
                pass
            try:
                writer.close()
            except Exception as e:
                logger.warning(f"Notice closing writer: {e}")
            # Clean up temporary audio file
            if temp_audio_file.exists():
                try:
                    temp_audio_file.unlink()
                except Exception:
                    pass
            clean_vram()

        logger.info(f"Video processing finished in {time.time() - start_time:.1f}s: {output_path}")
        return output_path
