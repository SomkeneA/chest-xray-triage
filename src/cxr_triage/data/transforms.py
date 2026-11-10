"""ImageNet-normalized transforms shared by training, evaluation, and the API."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _EnsureRGB:
    """NIH radiographs are grayscale. DenseNet expects three channels."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def build_transforms(image_size: int, *, train: bool) -> transforms.Compose:
    """Build the train or eval transform.

    Training uses a mild crop, a horizontal flip, a small rotation, and a small
    contrast jitter. Chest X-rays are not natural photographs, so the jitter
    stays narrow. Evaluation is a deterministic resize and center crop.
    """
    if image_size < 32:
        raise ValueError("image_size must be at least 32.")
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    resize_to = image_size + 32
    if train:
        return transforms.Compose(
            [
                _EnsureRGB(),
                transforms.Resize(resize_to),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            _EnsureRGB(),
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
