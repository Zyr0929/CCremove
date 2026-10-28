"""
Model Weights Downloader Script for SubClean.
Downloads pretrained models for ProPainter, RAFT optical flow, and LaMa inpainting.
"""

import os
import sys
from pathlib import Path
import urllib.request
import logging

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_weights")

# Model checkpoints repository URLs (verified live Hugging Face endpoints)
WEIGHTS_URLS = {
    "big-lama.pt": "https://huggingface.co/JosephCatrambone/big-lama-torchscript/resolve/main/lama.pt",
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
}


class DownloadProgressBar:
    def __init__(self, filename: str):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        if percent != self.last_percent and percent % 5 == 0:
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading {self.filename}: [{percent:3d}%] ({mb_down:.1f}/{mb_tot:.1f} MB)")
            sys.stdout.flush()
            self.last_percent = percent
        if downloaded >= total_size:
            sys.stdout.write(f"\rDownloading {self.filename}: [100%] Done!                                  \n")
            sys.stdout.flush()


def download_all(weights_dir: Path):
    weights_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory for model weights: {weights_dir.resolve()}")

    for filename, url in WEIGHTS_URLS.items():
        dest = weights_dir / filename
        if dest.exists() and dest.stat().st_size > 1024 * 100:
            logger.info(f"✓ Found existing weights: {filename} ({dest.stat().st_size / (1024*1024):.1f} MB)")
            continue

        logger.info(f"Fetching {filename} from {url}...")
        try:
            progress = DownloadProgressBar(filename)
            urllib.request.urlretrieve(url, str(dest), reporthook=progress)
            logger.info(f"✓ Successfully downloaded {filename} ({dest.stat().st_size / (1024*1024):.1f} MB)")
        except Exception as e:
            logger.warning(f"Could not download {filename} automatically ({e}).")
            logger.warning(f"You can manually download it from {url} and place it in {weights_dir.resolve()}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    w_dir = base_dir / "weights"
    download_all(w_dir)
