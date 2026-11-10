"""Grad-CAM returns a finite map aligned with the input."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from cxr_triage.explainability.gradcam import GradCAM, label_index, overlay_gradcam
from cxr_triage.labels import CHEST14_LABELS
from cxr_triage.models.densenet import build_model


def test_heatmap_matches_input_and_stays_in_unit_interval():
    model = build_model(num_classes=14, pretrained=False, dropout=0.0)
    image = torch.randn(1, 3, 64, 64)
    with GradCAM(model) as cam:
        heatmap = cam(image, class_index=CHEST14_LABELS.index("Effusion"))
    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0


def test_label_lookup_and_overlay():
    assert label_index("pleural thickening") == CHEST14_LABELS.index("Pleural_Thickening")
    with pytest.raises(ValueError, match="Unknown finding"):
        label_index("Not a finding")
    image = Image.new("L", (16, 16), color=40)
    heatmap = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    overlay = overlay_gradcam(image, heatmap)
    assert overlay.size == (16, 16)
    assert overlay.mode == "RGB"
