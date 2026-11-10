"""Class-imbalance weights."""

from __future__ import annotations

import numpy as np
import pytest

from cxr_triage.training.loss import compute_pos_weight


def test_pos_weight_uses_negative_over_positive_and_caps_rare_labels():
    targets = np.zeros((10, 3), dtype=np.float32)
    targets[:2, 0] = 1
    targets[:, 1] = 1
    capped = compute_pos_weight(targets, max_weight=3)
    assert capped[0].item() == pytest.approx(3.0)
    assert capped[1].item() == pytest.approx(0.0)
    assert capped[2].item() == pytest.approx(1.0)
    raw = compute_pos_weight(targets, max_weight=None)
    assert raw[0].item() == pytest.approx(4.0)


def test_pos_weight_rejects_an_empty_split():
    with pytest.raises(ValueError, match="empty"):
        compute_pos_weight(np.zeros((0, 14), dtype=np.float32))
