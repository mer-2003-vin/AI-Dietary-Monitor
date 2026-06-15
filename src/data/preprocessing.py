"""Image augmentation and normalization transforms for Food-101."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


# ImageNet mean/std used since EfficientNet was pretrained on ImageNet
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> A.Compose:
    # albumentations 2.0+ uses size=(h, w) instead of height/width
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    size_before_crop = int(image_size * 1.14)
    return A.Compose([
        A.Resize(height=size_before_crop, width=size_before_crop),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


class AlbumentationsWrapper:
    """Wraps albumentations transforms to work with PIL images (torchvision-style)."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image: Image.Image):
        image_np = np.array(image)
        result = self.transform(image=image_np)
        return result["image"]
