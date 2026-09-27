"""
Small helpers shared by training / evaluation / inference.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Device / reproducibility
# --------------------------------------------------------------------------- #
def get_device(device: str = "auto") -> torch.device:
    """Resolve 'auto' to the best available accelerator (MPS, CUDA or CPU)."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_device(device: torch.device) -> str:
    if device.type == "mps":
        return f"mps (Apple GPU, torch {torch.__version__})"
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)}, torch {torch.__version__})"
    return f"cpu ({os.cpu_count()} threads, torch {torch.__version__})"


# --------------------------------------------------------------------------- #
# Logging (stdout + file, unbuffered so that `tail -f` works while training)
# --------------------------------------------------------------------------- #
class Logger:
    """Minimal tee logger: writes every line to stdout and to a log file."""

    def __init__(self, log_file: Optional[str | Path] = None) -> None:
        self.file = None
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(log_file, "a", buffering=1, encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, message: str = "") -> None:
        stamp = f"[{human_time(time.time() - self.t0):>8}] "
        line = stamp + message if message else ""
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def human_time(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class AverageMeter:
    """Running mean of a scalar stream (losses)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


# --------------------------------------------------------------------------- #
# JSON / checkpoints
# --------------------------------------------------------------------------- #
def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    arch: Dict[str, Any],
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best: Optional[Dict[str, float]] = None,
    history: Optional[list] = None,
) -> None:
    """Checkpoint with everything needed to rebuild the model and resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "config": config,
        "epoch": epoch,
        "best": best or {},
        "history": history or [],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint, transparently accepting a raw ``state_dict`` as well.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    return {"state_dict": payload, "arch": {}, "config": {}, "epoch": None,
            "best": {}, "history": []}
