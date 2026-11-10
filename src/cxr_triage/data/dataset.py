"""PyTorch dataset for multi-label ChestX-ray14 images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cxr_triage.labels import CHEST14_LABELS, encode_labels


class ChestXrayDataset(Dataset):
    """Load one radiograph and a 14-dimensional multi-hot target.

    Images are converted to RGB because the ImageNet DenseNet expects three
    channels. The NIH files themselves are grayscale PNGs.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        transform: transforms.Compose | None,
        labels: tuple[str, ...] = CHEST14_LABELS,
    ) -> None:
        required = {"image_index", "finding_labels"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels = tuple(labels)
        self.targets = np.stack([encode_labels(value, self.labels) for value in self.frame["finding_labels"]])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        path = self.image_dir / str(row["image_index"])
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing radiograph {path}. Check image_dir or run the download script."
            ) from exc
        tensor = self.transform(rgb) if self.transform is not None else _to_tensor(rgb)
        target = torch.from_numpy(self.targets[index].copy())
        return tensor, target


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)
