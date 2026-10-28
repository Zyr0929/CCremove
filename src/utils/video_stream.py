"""
Streaming frame reader for video ingestion.
Reads video frames sequentially or in discrete chunks via generator
to prevent high RAM consumption on long or high-resolution videos.
"""

from pathlib import Path
from typing import Generator, List, Tuple, Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VideoStreamReader:
    """Streams video frames in memory-efficient chunks without loading entire video to RAM."""

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 25.0  # safe default fallback
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0

        logger.info(
            f"Opened video: {Path(self.video_path).name} "
            f"({self.width}x{self.height} @ {self.fps:.2f}fps, {self.total_frames} frames, {self.duration:.1f}s)"
        )

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """Fetch a specific frame by index (0-indexed) for interactive preview."""
        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame
        return None

    def read_frames(self, start_idx: int = 0, count: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (frame_index, frame_bgr) one frame at a time.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        current_idx = start_idx
        frames_yielded = 0

        while True:
            if count is not None and frames_yielded >= count:
                break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield current_idx, frame
            current_idx += 1
            frames_yielded += 1

    def read_chunks(self, chunk_size: int = 15) -> Generator[Tuple[List[int], List[np.ndarray]], None, None]:
        """
        Yields batches of frames: (list_of_indices, list_of_bgr_frames).
        Allows chunked batch inference for video inpainting without OOM crashes.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        chunk_indices = []
        chunk_frames = []
        current_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            chunk_indices.append(current_idx)
            chunk_frames.append(frame)
            current_idx += 1

            if len(chunk_frames) >= chunk_size:
                yield chunk_indices, chunk_frames
                chunk_indices = []
                chunk_frames = []

        if len(chunk_frames) > 0:
            yield chunk_indices, chunk_frames

    def close(self):
        """Release underlying OpenCV capture."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
