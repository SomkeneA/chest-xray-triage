"""FastAPI health and predict, including a Grad-CAM payload."""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from cxr_triage.api.app import Settings, create_app
from cxr_triage.labels import CHEST14_LABELS, CLINICAL_DISCLAIMER
from cxr_triage.models.checkpoint import save_checkpoint
from cxr_triage.models.densenet import build_model


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (48, 40), color=80).save(buffer, format="PNG")
    return buffer.getvalue()


def _checkpoint(path):
    model = build_model(num_classes=14, pretrained=False, dropout=0.0)
    save_checkpoint(
        path,
        model=model,
        labels=CHEST14_LABELS,
        image_size=64,
        dropout=0.0,
        epoch=0,
        score=None,
    )
    return path


def test_health_without_a_checkpoint():
    app = create_app(Settings(model_path=None, device="cpu"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": False}
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
        assert response.status_code == 503


def test_predict_returns_fourteen_probabilities_and_optional_gradcam(tmp_path):
    app = create_app(Settings(model_path=_checkpoint(tmp_path / "api.pt"), device="cpu"))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/predict",
            files={"file": ("x.png", _png(), "image/png")},
            params={"gradcam": "true", "label": "Effusion"},
        )
        assert response.status_code == 200
        body = response.json()
        assert [row["label"] for row in body["predictions"]] == list(CHEST14_LABELS)
        assert all(0.0 <= row["probability"] <= 1.0 for row in body["predictions"])
        assert body["gradcam_label"] == "Effusion"
        assert base64.b64decode(body["gradcam_png_base64"]).startswith(b"\x89PNG")
        assert body["disclaimer"] == CLINICAL_DISCLAIMER

        rejected = client.post(
            "/predict",
            files={"file": ("x.png", _png(), "image/png")},
            params={"label": "NotADisease"},
        )
        assert rejected.status_code == 400
        empty = client.post("/predict", files={"file": ("x.png", b"", "image/png")})
        assert empty.status_code == 400
        garbage = client.post("/predict", files={"file": ("x.png", b"not-an-image", "image/png")})
        assert garbage.status_code == 400
