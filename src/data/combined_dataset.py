"""
Combined Food-101 + Indian Food dataset.

Food-101:  101 classes, 750 train / 250 test per class
Indian:     80 classes,  50 images per class (40 train / 10 val split)

Final mapping: Food-101 classes 0-100, Indian classes 101-180 (181 total)
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


INDIAN_ROOT = Path("data/raw/indian-food/Indian Food Images/Indian Food Images")


def _get_indian_classes() -> list[str]:
    return sorted([d.name for d in INDIAN_ROOT.iterdir() if d.is_dir()])


def _build_combined_mapping(food101_classes_file: Path):
    with open(food101_classes_file) as f:
        food101_classes = [line.strip() for line in f if line.strip()]
    indian_classes = _get_indian_classes()

    # Merge: Food-101 first, then Indian (skip duplicates)
    food101_set = set(food101_classes)
    extra_indian = [c for c in indian_classes if c not in food101_set]

    all_classes = food101_classes + extra_indian
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


class CombinedFoodDataset(Dataset):
    """
    Merges Food-101 and Indian Food Images into one dataset.

    Indian food classes that share a name with Food-101 classes are
    merged into the same label (e.g. 'naan' already in Food-101 stays
    at its Food-101 index; new Indian classes get indices 101+).
    """

    def __init__(
        self,
        food101_dir: str,
        split: str = "train",
        transform=None,
        indian_val_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.transform = transform
        food101_dir = Path(food101_dir)

        self.class_to_idx, self.idx_to_class = _build_combined_mapping(
            food101_dir / "meta" / "classes.txt"
        )

        self.samples: list[tuple[str, int]] = []

        # ── Food-101 samples ──────────────────────────────────────────────────
        split_file = food101_dir / "meta" / f"{split}.txt"
        with open(split_file) as f:
            for line in f:
                img_rel = line.strip()           # e.g. "apple_pie/1234"
                class_name = img_rel.split("/")[0]
                full_path = food101_dir / "images" / f"{img_rel}.jpg"
                label = self.class_to_idx[class_name]
                self.samples.append((str(full_path), label))

        # ── Indian food samples ───────────────────────────────────────────────
        rng = random.Random(seed)
        for class_dir in sorted(INDIAN_ROOT.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            label = self.class_to_idx[class_name]
            images = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            rng.shuffle(images)
            n_val = max(1, int(len(images) * indian_val_fraction))
            val_imgs   = images[:n_val]
            train_imgs = images[n_val:]

            chosen = val_imgs if split == "test" else train_imgs
            for img_path in chosen:
                self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return len(self.class_to_idx)
