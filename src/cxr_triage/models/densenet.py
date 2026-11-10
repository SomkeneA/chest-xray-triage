"""DenseNet121 with a 14-logit multi-label head."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_model(
    num_classes: int = 14,
    *,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace the ImageNet classifier with a multi-label linear head.

    ``pretrained=True`` downloads ImageNet-1K weights on first use. Tests and
    the synthetic smoke config leave this off so they can run offline.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    if dropout > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.classifier = nn.Linear(in_features, num_classes)
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial normalization in torchvision's DenseNet feature stack.

    The forward pass applies an inplace ReLU after this layer. Grad-CAM clones
    the activations inside its hook so that inplace op cannot overwrite them.
    """
    layer = getattr(getattr(model, "features", None), "norm5", None)
    if layer is None:
        raise TypeError("Expected a torchvision DenseNet with features.norm5.")
    return layer
