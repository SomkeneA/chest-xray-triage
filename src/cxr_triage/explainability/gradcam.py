"""Grad-CAM for the DenseNet121 multi-label head.

The heatmap is a localization aid for review. It is not a segmentation and it
is not evidence that a finding is present.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cxr_triage.data.transforms import build_transforms
from cxr_triage.labels import CHEST14_LABELS, CLINICAL_DISCLAIMER
from cxr_triage.models.checkpoint import load_checkpoint
from cxr_triage.models.densenet import gradcam_target_layer

logger = logging.getLogger(__name__)


class GradCAM:
    """Capture the last spatial map in DenseNet and weight it by class gradients."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or gradcam_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _save_activation(self, module: torch.nn.Module, inputs, output: torch.Tensor) -> None:
        # Detach a copy before torchvision's trailing inplace ReLU can alias this tensor.
        self._activations = output.detach().clone()

    def _save_gradient(self, module: torch.nn.Module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach().clone()

    def __call__(self, image: torch.Tensor, class_index: int) -> np.ndarray:
        """Return a heatmap in ``[0, 1]`` with the same height and width as ``image``."""
        if image.ndim != 4:
            raise ValueError(f"Expected a batch shaped (1, C, H, W). Got {tuple(image.shape)}.")
        if class_index < 0:
            raise ValueError("class_index must be non-negative.")
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self._forward(image)
            if class_index >= logits.shape[1]:
                raise ValueError(f"class_index {class_index} is outside 0..{logits.shape[1] - 1}.")
            logits[:, class_index].sum().backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM did not capture the target layer. Check features.norm5.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = cam[0, 0]
        heatmap = heatmap - heatmap.min()
        maximum = heatmap.max()
        if float(maximum) > 0:
            heatmap = heatmap / maximum
        return heatmap.detach().cpu().numpy()

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        # Avoid torchvision's inplace ReLU, which breaks the backward hook.
        features = self.model.features(image)
        pooled = F.adaptive_avg_pool2d(F.relu(features, inplace=False), (1, 1))
        return self.model.classifier(torch.flatten(pooled, 1))


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet heatmap onto a radiograph. ``heatmap`` values lie in ``[0, 1]``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    cam = np.asarray(heatmap, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError("heatmap must be a 2-D array.")
    cam = np.clip(cam, 0.0, 1.0)
    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]), Image.Resampling.BILINEAR)
    color = (_colormap(cam) * 255).astype(np.uint8)
    blended = (1.0 - alpha) * np.asarray(base, dtype=np.float32) + alpha * color
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")


def label_index(name: str, labels: tuple[str, ...] = CHEST14_LABELS) -> int:
    """Resolve a finding name, accepting case differences and spaces for underscores."""
    query = name.strip().lower().replace(" ", "_")
    lookup = {label.lower(): index for index, label in enumerate(labels)}
    if query not in lookup:
        known = ", ".join(labels)
        raise ValueError(f"Unknown finding {name!r}. Expected one of: {known}")
    return lookup[query]


def generate_gradcam(
    checkpoint: str | Path,
    image_path: str | Path,
    label: str,
    output: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    """Write an overlay PNG for one image and one finding."""
    loaded = load_checkpoint(checkpoint, device)
    index = label_index(label, loaded.labels)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = build_transforms(loaded.image_size, train=False)(image).unsqueeze(0).to(device)
    with GradCAM(loaded.model) as cam:
        heatmap = cam(tensor, index)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay_gradcam(image, heatmap).save(destination)
    logger.info("Wrote Grad-CAM for %s to %s. %s", loaded.labels[index], destination, CLINICAL_DISCLAIMER)
    return destination


def _colormap(cam: np.ndarray) -> np.ndarray:
    """Jet-style RGB in 0..1 without requiring a display."""
    x = np.clip(cam, 0.0, 1.0)
    red = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0, 1)
    green = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0, 1)
    blue = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0, 1)
    return np.stack([red, green, blue], axis=-1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a Grad-CAM overlay for one radiograph.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True, help="Finding name, for example Cardiomegaly.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_gradcam(args.checkpoint, args.image, args.label, args.output, device=args.device)


if __name__ == "__main__":
    main()
