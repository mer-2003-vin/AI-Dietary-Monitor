"""
Food-101 dataset loader.

Download: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
Extract to: data/raw/food-101/
Expected structure:
  data/raw/food-101/
    images/
      apple_pie/  pizza/  ...
    meta/
      train.txt  test.txt  classes.txt
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._load_meta()

    def _load_meta(self):
        classes_file = self.root_dir / "meta" / "classes.txt"
        split_file = self.root_dir / "meta" / f"{self.split}.txt"

        with open(classes_file) as f:
            classes = [line.strip() for line in f]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        with open(split_file) as f:
            for line in f:
                img_path = line.strip()          # e.g. "apple_pie/1234"
                class_name = img_path.split("/")[0]
                full_path = self.root_dir / "images" / f"{img_path}.jpg"
                label = self.class_to_idx[class_name]
                self.samples.append((str(full_path), label))

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
