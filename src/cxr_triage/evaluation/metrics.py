"""Per-label AUROC, sensitivity, specificity, and F1."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from cxr_triage.evaluation.calibration import expected_calibration_error


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """AUROC, or None when the column contains a single class."""
    labels = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(y_prob).astype(float).ravel()
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def mean_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Macro AUROC over labels that contain both classes."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scores = [
        score
        for index in range(y_true.shape[1])
        if (score := safe_auroc(y_true[:, index], y_prob[:, index])) is not None
    ]
    if not scores:
        return None
    return float(np.mean(scores))


def operating_point(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Sensitivity, specificity, and F1 at one threshold.

    Sensitivity and F1 are None when there are no positive studies. Specificity
    is None when there are no negative studies. Means skip those undefined values.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(y_prob).astype(float).ravel()
    predicted = (scores >= threshold).astype(int)
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    true_negative = int(((predicted == 0) & (labels == 0)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    sensitivity = _rate(true_positive, true_positive + false_negative)
    specificity = _rate(true_negative, true_negative + false_positive)
    precision = _rate(true_positive, true_positive + false_positive)
    if sensitivity is None:
        f1 = None
    elif precision is None or precision + sensitivity == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "support": int(labels.sum()),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Pick a threshold on a validation split. Do not fit this on the test split."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in grid:
        f1 = operating_point(y_true, y_prob, float(threshold))["f1"]
        score = -1.0 if f1 is None else float(f1)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    labels: Sequence[str],
    thresholds: Sequence[float] | float,
    ece_bins: int = 15,
) -> dict:
    """Build the JSON-ready metric report for one split."""
    labels = list(labels)
    truth = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(y_prob, dtype=np.float64)
    if truth.shape != probs.shape or truth.ndim != 2 or truth.shape[1] != len(labels):
        raise ValueError(
            f"Expected predictions shaped (N, {len(labels)}). "
            f"Got truth {truth.shape} and probs {probs.shape}."
        )
    threshold_array = _broadcast_thresholds(thresholds, len(labels))
    per_label: dict[str, dict] = {}
    aurocs: list[float] = []
    sensitivities: list[float] = []
    specificities: list[float] = []
    f1_scores: list[float] = []
    eces: list[float] = []
    for index, name in enumerate(labels):
        point = operating_point(truth[:, index], probs[:, index], float(threshold_array[index]))
        auroc = safe_auroc(truth[:, index], probs[:, index])
        ece = expected_calibration_error(truth[:, index], probs[:, index], n_bins=ece_bins)
        per_label[name] = {
            "auroc": auroc,
            "sensitivity": point["sensitivity"],
            "specificity": point["specificity"],
            "f1": point["f1"],
            "threshold": float(threshold_array[index]),
            "support": point["support"],
            "ece": None if math.isnan(ece) else ece,
        }
        _collect(aurocs, auroc)
        _collect(sensitivities, point["sensitivity"])
        _collect(specificities, point["specificity"])
        _collect(f1_scores, point["f1"])
        if not math.isnan(ece):
            eces.append(ece)
    micro_ece = expected_calibration_error(truth.ravel(), probs.ravel(), n_bins=ece_bins)
    return {
        "num_images": int(truth.shape[0]),
        "mean_auroc": _mean(aurocs),
        "mean_sensitivity": _mean(sensitivities),
        "mean_specificity": _mean(specificities),
        "mean_f1": _mean(f1_scores),
        "mean_ece": _mean(eces),
        "micro_ece": None if math.isnan(micro_ece) else micro_ece,
        "per_label": per_label,
        "notes": [
            "AUROC is omitted for a label when that split contains only one class.",
            "Sensitivity and F1 are omitted when the label has no positive studies.",
            "Specificity is omitted when the label has no negative studies.",
            "Means use only the labels where that metric is defined.",
            "These numbers describe the split that was evaluated. They are not a clinical claim.",
        ],
    }


def _broadcast_thresholds(thresholds: Sequence[float] | float, count: int) -> np.ndarray:
    if isinstance(thresholds, (int, float)):
        return np.full(count, float(thresholds), dtype=np.float64)
    array = np.asarray(list(thresholds), dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"Expected {count} thresholds, got {array.shape}.")
    return array


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _collect(bucket: list[float], value: float | int | None) -> None:
    if value is not None:
        bucket.append(float(value))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))
