"""Run a model over a loader and stack probabilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(targets, probabilities)`` with shape ``(N, C)``."""
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, batch_targets in loader:
        logits = model(images.to(device))
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(batch_targets.numpy())
    if not targets:
        raise ValueError("The data loader produced no batches.")
    return np.concatenate(targets), np.concatenate(probabilities)
