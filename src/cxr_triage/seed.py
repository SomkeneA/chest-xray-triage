"""Reproducible seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. CUDA convolution is still nondeterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda``.

    ``auto`` uses a GPU when one is visible. An explicit ``cuda`` request fails
    loudly instead of silently training on CPU.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is set to cuda, but no GPU is visible. Use auto or cpu.")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown device {name!r}. Expected auto, cpu, or cuda.")
    return torch.device(name)
