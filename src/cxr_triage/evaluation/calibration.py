"""Expected calibration error and reliability diagrams.

ECE bins predicted probabilities and compares the mean confidence in each bin
with the fraction of those studies that are positive. A smaller error means
the probabilities are closer to observed frequencies.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[list[float], list[float], list[int]]:
    """Return bin confidence, observed frequency, and count for non-empty bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    labels = np.asarray(y_true, dtype=np.float64).ravel()
    probs = np.asarray(y_prob, dtype=np.float64).ravel()
    if labels.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same shape.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confidences: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidences.append(float(probs[mask].mean()))
        accuracies.append(float(labels[mask].mean()))
        counts.append(count)
    return confidences, accuracies, counts


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Weighted mean absolute gap between confidence and observed frequency."""
    confidences, accuracies, counts = reliability_curve(y_true, y_prob, n_bins=n_bins)
    total = sum(counts)
    if total == 0:
        return float("nan")
    error = 0.0
    for confidence, accuracy, count in zip(confidences, accuracies, counts, strict=True):
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def save_reliability_diagrams(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] | tuple[str, ...],
    path: str | Path,
    n_bins: int = 15,
) -> None:
    """Write one reliability curve per finding."""
    import matplotlib.pyplot as plt

    labels = list(labels)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(labels):
        raise ValueError("y_true, y_prob, and labels do not line up.")

    columns = 4
    rows = int(np.ceil(len(labels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.1))
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, name, index in zip(flat_axes, labels, range(len(labels)), strict=False):
        confidences, accuracies, _counts = reliability_curve(
            y_true[:, index], y_prob[:, index], n_bins=n_bins
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
        if confidences:
            axis.plot(confidences, accuracies, marker="o", linewidth=1.4)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.set_xlabel("Mean confidence")
        axis.set_ylabel("Observed frequency")
    for axis in flat_axes[len(labels) :]:
        axis.axis("off")
    figure.suptitle("Reliability diagrams", fontsize=12)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
