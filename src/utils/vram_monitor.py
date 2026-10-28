"""
Memory and VRAM monitoring utilities.
Tracks GPU VRAM, system RAM, and handles automatic cache clearing to prevent OOM errors.
"""

import gc
import logging
from typing import Dict, Any
import psutil
import torch

logger = logging.getLogger(__name__)


def clean_vram():
    """Forces garbage collection and releases cached PyTorch CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class VRAMMonitor:
    """Monitors real-time GPU VRAM and host RAM usage."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = (device == "cuda" and torch.cuda.is_available())

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
        - vram_used_gb: Current allocated VRAM in GB
        - vram_reserved_gb: Current reserved VRAM in GB
        - vram_total_gb: Total physical VRAM in GB
        - vram_percent: Percentage of VRAM reserved
        - ram_used_gb: Current system RAM used in GB
        - ram_total_gb: Total system RAM in GB
        - ram_percent: Percentage of system RAM used
        """
        sys_ram = psutil.virtual_memory()
        stats = {
            "ram_used_gb": round(sys_ram.used / (1024 ** 3), 2),
            "ram_total_gb": round(sys_ram.total / (1024 ** 3), 2),
            "ram_percent": sys_ram.percent,
            "device": self.device,
        }

        if self.is_cuda:
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(dev_idx)
            reserved = torch.cuda.memory_reserved(dev_idx)

            stats.update({
                "gpu_name": props.name,
                "vram_allocated_gb": round(allocated / (1024 ** 3), 2),
                "vram_reserved_gb": round(reserved / (1024 ** 3), 2),
                "vram_total_gb": round(total / (1024 ** 3), 2),
                "vram_percent": round((reserved / total) * 100, 1),
            })
        else:
            stats.update({
                "gpu_name": "CPU / System Memory",
                "vram_allocated_gb": 0.0,
                "vram_reserved_gb": 0.0,
                "vram_total_gb": stats["ram_total_gb"],
                "vram_percent": stats["ram_percent"],
            })

        return stats

    def format_summary(self) -> str:
        """Returns a formatted one-line memory summary string."""
        s = self.get_memory_stats()
        if self.is_cuda:
            return (
                f"GPU: {s['gpu_name']} | "
                f"VRAM: {s['vram_reserved_gb']:.2f}/{s['vram_total_gb']:.2f} GB ({s['vram_percent']}%) | "
                f"RAM: {s['ram_used_gb']:.2f}/{s['ram_total_gb']:.2f} GB"
            )
        return f"System RAM: {s['ram_used_gb']:.2f}/{s['ram_total_gb']:.2f} GB ({s['ram_percent']}%)"
