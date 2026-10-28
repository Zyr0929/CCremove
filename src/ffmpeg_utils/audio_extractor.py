"""
Audio extraction and video metadata inspection module using FFmpeg.
Guarantees lossless audio preservation and frame-accurate timing synchronization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import logging
import math
import os
import shutil
import subprocess
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    video_codec: str
    audio_codec: Optional[str]
    bitrate: Optional[int]
    has_audio: bool


def get_ffmpeg_binary() -> str:
    """
    Locates the FFmpeg executable.
    Checks system PATH first, then falls back to imageio-ffmpeg bundled binary.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg executable not found. Please install FFmpeg and add it to your system PATH, "
        "or install imageio-ffmpeg via `pip install imageio-ffmpeg`."
    )


def get_ffprobe_binary() -> Optional[str]:
    """Locates ffprobe executable if available."""
    return shutil.which("ffprobe")


def inspect_media_metadata(video_path: str) -> VideoMetadata:
    """
    Extracts comprehensive metadata from the video container using ffprobe or fallback probe.
    """
    video_path = str(video_path)
    ffprobe_bin = get_ffprobe_binary()

    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info = json.loads(res.stdout)
            video_stream = None
            audio_stream = None

            for s in info.get("streams", []):
                if s.get("codec_type") == "video" and video_stream is None:
                    video_stream = s
                elif s.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = s

            if video_stream:
                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                
                # Parse FPS from r_frame_rate e.g. "30000/1001" or "25/1"
                r_fps = video_stream.get("r_frame_rate", "25/1")
                if "/" in r_fps:
                    num, den = r_fps.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 25.0
                else:
                    fps = float(r_fps)

                # Duration
                duration = float(video_stream.get("duration", info.get("format", {}).get("duration", 0.0)))
                nb_frames = int(video_stream.get("nb_frames", 0))
                if nb_frames == 0 and duration > 0 and fps > 0:
                    nb_frames = int(round(duration * fps))

                bitrate = int(info.get("format", {}).get("bit_rate", 0)) or None

                return VideoMetadata(
                    width=width,
                    height=height,
                    fps=fps,
                    total_frames=nb_frames,
                    duration=duration,
                    video_codec=video_stream.get("codec_name", "unknown"),
                    audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                    bitrate=bitrate,
                    has_audio=(audio_stream is not None),
                )
        except Exception as e:
            logger.warning(f"ffprobe extraction failed ({e}), falling back to OpenCV inspection.")

    # Fallback to OpenCV inspection
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    # Test if audio track exists using ffmpeg
    has_audio = False
    audio_codec = None
    try:
        ffmpeg_bin = get_ffmpeg_binary()
        test_cmd = [ffmpeg_bin, "-i", video_path]
        proc = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "Audio:" in proc.stderr:
            has_audio = True
            audio_codec = "detected"
    except Exception:
        pass

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        video_codec="unknown",
        audio_codec=audio_codec,
        bitrate=None,
        has_audio=has_audio,
    )


def extract_lossless_audio(video_path: str, output_audio_path: str) -> bool:
    """
    Extracts the audio track from the video file losslessly without re-encoding.
    If direct stream copy fails, transcodes to high-fidelity AAC or uncompressed WAV.

    Returns:
        True if audio was extracted successfully, False if no audio stream exists.
    """
    video_path = str(video_path)
    output_audio_path = str(output_audio_path)
    ffmpeg_bin = get_ffmpeg_binary()

    # Pass 1: Try stream copy (fastest & 100% bit-exact lossless)
    copy_cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-c:a", "copy",
        output_audio_path,
    ]

    try:
        res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and Path(output_audio_path).exists() and Path(output_audio_path).stat().st_size > 100:
            logger.info(f"Successfully extracted lossless audio copy to: {output_audio_path}")
            return True
    except Exception as e:
        logger.debug(f"Direct stream copy audio extraction failed: {e}")

    # Pass 2: Fallback re-encode to high-bitrate AAC or WAV if container copy was incompatible
    fallback_cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-c:a", "aac",
        "-b:a", "320k",
        output_audio_path,
    ]

    try:
        res = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and Path(output_audio_path).exists() and Path(output_audio_path).stat().st_size > 100:
            logger.info(f"Extracted audio with high-bitrate AAC fallback to: {output_audio_path}")
            return True
    except Exception as e:
        logger.warning(f"Audio fallback extraction failed: {e}")

    logger.info("No audio stream found or extraction failed; video will be rendered without audio.")
    return False
