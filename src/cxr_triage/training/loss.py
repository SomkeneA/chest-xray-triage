"""Positive-class weights for multi-label BCE-with-logits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(targets: np.ndarray, max_weight: float | None = None) -> torch.Tensor:
    """Return per-label ``(n_negative / n_positive)`` weights.

    A label with no positives gets weight 1, because the ratio is undefined.
    ``max_weight`` caps rare findings such as hernia so they cannot dominate
    the loss. Weights are computed from the training split only.
    """
    if targets.ndim != 2:
        raise ValueError(f"Expected targets shaped (N, C), got {targets.shape}.")
    if targets.shape[0] == 0:
        raise ValueError("Cannot compute pos_weight from an empty training set.")
    positives = targets.sum(axis=0).astype(np.float64)
    negatives = targets.shape[0] - positives
    weights = np.ones(targets.shape[1], dtype=np.float64)
    observed = positives > 0
    weights[observed] = negatives[observed] / positives[observed]
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def build_loss(pos_weight: torch.Tensor | None) -> nn.Module:
    """Multi-label loss. ``pos_weight`` is applied per finding."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
