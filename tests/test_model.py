"""DenseNet head shape and checkpoint reload."""

from __future__ import annotations

import torch

from cxr_triage.labels import CHEST14_LABELS
from cxr_triage.models.checkpoint import load_checkpoint, save_checkpoint
from cxr_triage.models.densenet import build_model


def test_densenet_outputs_one_logit_per_finding():
    model = build_model(num_classes=14, pretrained=False, dropout=0.1)
    assert isinstance(model.classifier, torch.nn.Sequential)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 14)


def test_checkpoint_roundtrip_reloads_the_same_logits(tmp_path):
    model = build_model(num_classes=14, pretrained=False, dropout=0.0)
    model.eval()
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        labels=CHEST14_LABELS,
        image_size=64,
        dropout=0.0,
        epoch=2,
        score=None,
    )
    loaded = load_checkpoint(path, "cpu")
    assert loaded.labels == CHEST14_LABELS
    assert loaded.image_size == 64
    batch = torch.zeros(1, 3, 64, 64)
    with torch.inference_mode():
        original = model(batch)
        restored = loaded.model(batch)
    assert torch.allclose(original, restored)
