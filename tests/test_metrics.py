"""Operating points, AUROC, and calibration on known arrays."""

from __future__ import annotations

import numpy as np
import pytest

from cxr_triage.evaluation.calibration import (
    expected_calibration_error,
    save_reliability_diagrams,
)
from cxr_triage.evaluation.metrics import (
    evaluate_predictions,
    operating_point,
    safe_auroc,
    select_f1_threshold,
)


def test_auroc_perfect_reversed_and_undefined():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert safe_auroc(labels, scores) == pytest.approx(1.0)
    assert safe_auroc(labels, 1 - scores) == pytest.approx(0.0)
    assert safe_auroc(np.ones(3), np.array([0.2, 0.4, 0.6])) is None


def test_sensitivity_specificity_and_f1_at_threshold():
    point = operating_point(
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.4, 0.8, 0.1]),
        0.5,
    )
    assert point["sensitivity"] == pytest.approx(0.5)
    assert point["specificity"] == pytest.approx(0.5)
    assert point["f1"] == pytest.approx(0.5)
    assert point["support"] == 2


def test_f1_threshold_search_finds_a_perfect_cut():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.6, 0.6, 0.6, 0.9])
    chosen = select_f1_threshold(labels, scores)
    assert operating_point(labels, scores, chosen)["f1"] == pytest.approx(1.0)


def test_ece_for_perfect_and_shifted_confidence():
    perfect = expected_calibration_error(np.ones(10), np.ones(10), n_bins=10)
    shifted = expected_calibration_error(np.ones(10), np.full(10, 0.9), n_bins=10)
    assert perfect == pytest.approx(0.0)
    assert shifted == pytest.approx(0.1)


def test_report_skips_undefined_auroc_in_the_mean():
    truth = np.array([[0, 1], [1, 1], [0, 1], [1, 1]], dtype=float)
    probs = np.array([[0.1, 0.2], [0.9, 0.3], [0.2, 0.4], [0.8, 0.5]], dtype=float)
    report = evaluate_predictions(truth, probs, labels=["A", "B"], thresholds=0.5, ece_bins=5)
    assert report["per_label"]["A"]["auroc"] == pytest.approx(1.0)
    assert report["per_label"]["B"]["auroc"] is None
    assert report["mean_auroc"] == pytest.approx(1.0)
    assert report["num_images"] == 4


def test_reliability_diagram_is_written(tmp_path):
    truth = np.array([[0, 1], [1, 0]], dtype=float)
    probs = np.array([[0.2, 0.8], [0.7, 0.1]], dtype=float)
    path = tmp_path / "reliability.png"
    save_reliability_diagrams(truth, probs, ["A", "B"], path, n_bins=5)
    assert path.is_file()
    assert path.stat().st_size > 0
