"""HTTP API for multi-label probabilities and an optional Grad-CAM image.

Endpoints:
- ``GET /health`` always reports that the process is up.
- ``GET /ready`` reports whether a checkpoint is loaded.
- ``POST /predict`` accepts one radiograph and returns 14 probabilities.

The process starts even when ``MODEL_PATH`` is missing, so a container health
check can pass before weights are mounted. ``/predict`` then returns 503.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from cxr_triage.data.transforms import build_transforms
from cxr_triage.explainability.gradcam import GradCAM, label_index, overlay_gradcam
from cxr_triage.labels import CLINICAL_DISCLAIMER
from cxr_triage.models.checkpoint import LoadedModel, load_checkpoint
from cxr_triage.seed import resolve_device

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class LabelProbability(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predictions: list[LabelProbability]
    gradcam_label: str | None = None
    gradcam_png_base64: str | None = None
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@dataclass
class Settings:
    model_path: Path | None = None
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> Settings:
        raw_path = os.environ.get("MODEL_PATH", "").strip()
        return cls(
            model_path=Path(raw_path) if raw_path else None,
            device=os.environ.get("CXR_DEVICE", "cpu").strip() or "cpu",
        )


@dataclass
class Runtime:
    settings: Settings
    device: torch.device
    loaded: LoadedModel | None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Pass ``settings`` in tests; the process uses the environment."""
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Chest X-ray triage",
        version="0.1.0",
        description=CLINICAL_DISCLAIMER,
    )
    app.state.runtime = _load_runtime(settings)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=app.state.runtime.loaded is not None)

    @app.get("/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        if app.state.runtime.loaded is None:
            raise HTTPException(status_code=503, detail="Model checkpoint is not loaded.")
        return HealthResponse(status="ready", model_loaded=True)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        file: UploadFile = File(...),
        gradcam: bool = Query(False, description="Include a Grad-CAM PNG, base64 encoded."),
        label: str | None = Query(
            None,
            description="Finding for Grad-CAM. Defaults to the highest probability.",
        ),
    ) -> PredictResponse:
        runtime: Runtime = app.state.runtime
        if runtime.loaded is None:
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint is not loaded. Set MODEL_PATH to a checkpoint from training.",
            )
        image = await _read_image(file)
        probabilities = _probabilities(runtime, image)
        chosen_label = None
        encoded = None
        if gradcam or label:
            chosen_label = label or runtime.loaded.labels[int(np.argmax(probabilities))]
            try:
                index = label_index(chosen_label, runtime.loaded.labels)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            chosen_label = runtime.loaded.labels[index]
            if gradcam:
                encoded = _gradcam_base64(runtime, image, index)
        return PredictResponse(
            predictions=[
                LabelProbability(label=name, probability=round(float(probability), 6))
                for name, probability in zip(runtime.loaded.labels, probabilities, strict=True)
            ],
            gradcam_label=chosen_label,
            gradcam_png_base64=encoded,
            disclaimer=CLINICAL_DISCLAIMER,
        )

    return app


def _load_runtime(settings: Settings) -> Runtime:
    device = resolve_device(settings.device)
    if settings.model_path is None:
        logger.warning("MODEL_PATH is not set. /health will pass and /predict will return 503.")
        return Runtime(settings=settings, device=device, loaded=None)
    if not settings.model_path.is_file():
        logger.warning("MODEL_PATH %s does not exist yet.", settings.model_path)
        return Runtime(settings=settings, device=device, loaded=None)
    loaded = load_checkpoint(settings.model_path, device)
    logger.info("Loaded checkpoint %s on %s", settings.model_path, device)
    return Runtime(settings=settings, device=device, loaded=loaded)


async def _read_image(file: UploadFile) -> Image.Image:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The upload is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB.")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The upload is not a readable image.") from exc


def _probabilities(runtime: Runtime, image: Image.Image) -> np.ndarray:
    assert runtime.loaded is not None
    tensor = build_transforms(runtime.loaded.image_size, train=False)(image)
    batch = tensor.unsqueeze(0).to(runtime.device)
    with torch.inference_mode():
        runtime.loaded.model.eval()
        logits = runtime.loaded.model(batch)
        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
    if probabilities.shape != (len(runtime.loaded.labels),):
        raise HTTPException(status_code=500, detail="The model did not return one probability per label.")
    if not np.isfinite(probabilities).all():
        raise HTTPException(status_code=500, detail="The model returned a non-finite probability.")
    return probabilities


def _gradcam_base64(runtime: Runtime, image: Image.Image, class_index: int) -> str:
    assert runtime.loaded is not None
    tensor = build_transforms(runtime.loaded.image_size, train=False)(image).unsqueeze(0)
    tensor = tensor.to(runtime.device)
    with GradCAM(runtime.loaded.model) as cam:
        heatmap = cam(tensor, class_index)
    overlay = overlay_gradcam(image, heatmap)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("cxr_triage.api.app:app", host=host, port=port)


app = create_app()
