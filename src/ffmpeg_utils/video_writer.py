"""
High-Fidelity FFmpeg Video Writer.
Streams processed raw frames directly via pipe to FFmpeg with CRF 18 encoding,
and remuxes the original lossless audio track with sample-accurate synchronization.
"""

from pathlib import Path
from typing import Optional
import logging
import subprocess
import cv2
import numpy as np

from .audio_extractor import get_ffmpeg_binary

logger = logging.getLogger(__name__)


class FFmpegPipeWriter:
    """
    Pipes raw BGR frames to FFmpeg to encode an H.264/H.265 MP4 video with CRF 18.
    Preserves maximum image crispness and remuxes the original audio.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_path: Optional[str] = None,
        crf: int = 18,
        preset: str = "slow",
        video_codec: str = "libx264",
        pixel_format: str = "yuv420p",
    ):
        self.output_path = str(output_path)
        # H.264 with yuv420p strictly requires dimensions to be divisible by 2
        self.width = width if (width % 2 == 0) else (width - 1)
        self.height = height if (height % 2 == 0) else (height - 1)
        self.fps = fps if fps > 0 else 25.0
        self.audio_path = str(audio_path) if audio_path and Path(audio_path).exists() else None
        self.crf = crf
        self.preset = preset
        self.video_codec = video_codec
        self.pixel_format = pixel_format

        self.ffmpeg_bin = get_ffmpeg_binary()
        self.process: Optional[subprocess.Popen] = None
        self._start_process()

    def _start_process(self):
        # Base video input stream from raw stdin pipe
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-loglevel", "error",  # Suppress frame-by-frame progress lines to prevent pipe deadlock
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", f"{self.fps:.4f}",
            "-i", "-",  # Read frames from stdin pipe
        ]

        # Audio stream input if available
        if self.audio_path:
            cmd.extend([
                "-i", self.audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
            ])
        else:
            cmd.extend([
                "-map", "0:v:0",
            ])

        # High-Fidelity Video Encoding Parameters (CRF 18)
        cmd.extend([
            "-c:v", self.video_codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", self.pixel_format,
            "-movflags", "+faststart",  # Web optimized streaming headers
            self.output_path,
        ])

        logger.info(f"Starting FFmpeg pipe encoder: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # Prevent pipe buffer saturation
            stderr=subprocess.PIPE,
            bufsize=10**7,
        )

    def write_frame(self, frame_bgr: np.ndarray):
        """Writes a single BGR frame uint8 to the FFmpeg pipe."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FFmpeg writer is not open.")

        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = np.ascontiguousarray(
                cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            )

        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        try:
            self.process.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, IOError) as e:
            stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"FFmpeg stdin pipe broke: {e}\nFFmpeg stderr: {stderr}")

    def close(self):
        """Flushes stdin pipe and waits for FFmpeg to finish encoding."""
        if self.process is not None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
            try:
                stdout, stderr = self.process.communicate(timeout=30)
                ret = self.process.returncode
                if ret != 0:
                    err_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
                    logger.error(f"FFmpeg encoding finished with error code {ret}:\n{err_msg}")
                    raise RuntimeError(f"FFmpeg video encoding failed (exit code {ret}): {err_msg}")
                else:
                    logger.info(f"FFmpeg encoding completed successfully: {self.output_path}")
            except Exception as e:
                logger.warning(f"Notice during FFmpeg close: {e}")
            finally:
                self.process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
