"""Checkpoint payload used by training, evaluation, and the API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cxr_triage.labels import CHEST14_LABELS
from cxr_triage.models.densenet import build_model


@dataclass
class LoadedModel:
    model: nn.Module
    labels: tuple[str, ...]
    image_size: int
    dropout: float
    epoch: int | None
    score: float | None
    config: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    labels: tuple[str, ...] | list[str],
    image_size: int,
    dropout: float,
    epoch: int,
    score: float | None,
    config: dict[str, Any] | None = None,
    pos_weight: list[float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write a training checkpoint. Weights are not committed to git."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "labels": list(labels),
        "image_size": int(image_size),
        "dropout": float(dropout),
        "epoch": int(epoch),
        "score": None if score is None else float(score),
        "config": config or {},
        "pos_weight": pos_weight,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, destination)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> LoadedModel:
    """Rebuild DenseNet121 and load weights saved by :func:`save_checkpoint`."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path} is not a checkpoint dictionary.")

    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        labels = tuple(payload.get("labels", CHEST14_LABELS))
        image_size = int(payload.get("image_size", 224))
        dropout = float(payload.get("dropout", 0.0))
        epoch = payload.get("epoch")
        score = payload.get("score")
        config = payload.get("config") or {}
    else:
        state = payload
        labels = CHEST14_LABELS
        image_size = 224
        dropout = _dropout_from_state(state)
        epoch = None
        score = None
        config = {}

    model = build_model(num_classes=len(labels), pretrained=False, dropout=dropout)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        labels=labels,
        image_size=image_size,
        dropout=dropout,
        epoch=None if epoch is None else int(epoch),
        score=None if score is None else float(score),
        config=dict(config),
    )


def _dropout_from_state(state: dict[str, Any]) -> float:
    """Raw state dicts use a plain linear head unless the sequential keys exist."""
    if any(str(key).startswith("classifier.1.") for key in state):
        return 0.1
    return 0.0
