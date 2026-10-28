"""FFmpeg utilities for lossless audio extraction and high-quality video encoding."""
from .audio_extractor import get_ffmpeg_binary, extract_lossless_audio, inspect_media_metadata, VideoMetadata
from .video_writer import FFmpegPipeWriter

__all__ = [
    "get_ffmpeg_binary",
    "extract_lossless_audio",
    "inspect_media_metadata",
    "VideoMetadata",
    "FFmpegPipeWriter",
]
