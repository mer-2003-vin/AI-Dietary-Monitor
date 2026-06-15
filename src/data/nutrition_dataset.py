"""
USDA FoodData Central nutrition dataset loader.

Download the CSV from:
  https://fdc.nal.usda.gov/download-datasets.html
  → "Full Download of All Data Types" → foundation_foods or sr_legacy_foods CSV

Expected: data/raw/usda_nutrition.csv
Required columns: description, calories, protein_g, carbs_g, fat_g,
                  fiber_g, sugar_g, sodium_mg, iron_mg, calcium_mg, vitamin_c_mg
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib


NUTRIENT_COLS = [
    "calories", "protein_g", "carbs_g", "fat_g",
    "fiber_g", "sugar_g", "sodium_mg", "iron_mg", "calcium_mg", "vitamin_c_mg"
]

TARGET_COLS = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]


class NutritionDataset(Dataset):
    """
    Maps food class index (from Food-101) → macro/calorie targets.
    Used to train the calorie estimator model.
    """

    def __init__(self, csv_path: str, food101_classes: list, scaler_path: str = None, fit_scaler: bool = False):
        self.df = self._load_and_align(csv_path, food101_classes)
        self.food101_classes = food101_classes

        self.X = np.eye(len(food101_classes), dtype=np.float32)  # one-hot class vectors
        self.y = self.df[TARGET_COLS].values.astype(np.float32)

        self.scaler = StandardScaler()
        if fit_scaler:
            self.y = self.scaler.fit_transform(self.y)
            if scaler_path:
                joblib.dump(self.scaler, scaler_path)
        elif scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            self.y = self.scaler.transform(self.y)

    def _load_and_align(self, csv_path: str, food101_classes: list) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.dropna(subset=TARGET_COLS)

        # Match Food-101 class names to USDA descriptions via fuzzy keyword match
        rows = []
        for cls in food101_classes:
            keyword = cls.replace("_", " ").lower()
            match = df[df["description"].str.lower().str.contains(keyword, na=False)]
            if len(match) > 0:
                row = match.iloc[0][TARGET_COLS].values
            else:
                row = [0.0] * len(TARGET_COLS)  # fallback zeros
            rows.append(row)

        return pd.DataFrame(rows, columns=TARGET_COLS)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class DailyIntakeDataset(Dataset):
    """
    Time-series dataset of daily nutritional intake for LSTM training.
    Each sample: sequence of 14 days → predict next day's intake.
    """

    def __init__(self, records: pd.DataFrame, seq_len: int = 14, nutrient_cols: list = None):
        self.seq_len = seq_len
        self.cols = nutrient_cols or NUTRIENT_COLS
        values = records[self.cols].values.astype(np.float32)
        self.scaler = StandardScaler()
        values = self.scaler.fit_transform(values)
        self.sequences, self.targets = self._build_sequences(values)

    def _build_sequences(self, values: np.ndarray):
        X, y = [], []
        for i in range(len(values) - self.seq_len):
            X.append(values[i: i + self.seq_len])
            y.append(values[i + self.seq_len])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])
