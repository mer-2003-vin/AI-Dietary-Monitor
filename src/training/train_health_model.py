"""
Train all health analytics models:
  1. DeficiencyDetector  (RandomForest)
  2. NutritionTrendLSTM  (LSTM)
  3. AnomalyDetector     (Isolation Forest)

Usage:
  python -m src.training.train_health_model

Requires: data/processed/daily_logs.csv
  Columns: user_id, date, calories, protein_g, carbs_g, fat_g,
           fiber_g, sugar_g, sodium_mg, iron_mg, calcium_mg, vitamin_c_mg
"""

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.models.health_analytics import (
    DeficiencyDetector, NutritionTrendLSTM, AnomalyDetector, NUTRIENT_COLS, DAILY_RDA
)
from src.data.nutrition_dataset import DailyIntakeDataset


def generate_synthetic_logs(n_users: int = 100, days: int = 90) -> pd.DataFrame:
    """
    Generates synthetic daily nutrition logs for training when real data is unavailable.
    Replace with real meal log data from the database in production.
    """
    np.random.seed(42)
    records = []
    rda = np.array([DAILY_RDA[n] for n in NUTRIENT_COLS])
    for user_id in range(n_users):
        for day in range(days):
            noise = np.random.normal(1.0, 0.3, size=len(NUTRIENT_COLS))
            intake = (rda * noise).clip(min=0)
            records.append({"user_id": user_id, "day": day, **dict(zip(NUTRIENT_COLS, intake))})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Train DeficiencyDetector
# ---------------------------------------------------------------------------

def train_deficiency_detector(df: pd.DataFrame, cfg: dict):
    window = 7  # 7-day window to compute average % RDA
    rda = np.array([DAILY_RDA[n] for n in NUTRIENT_COLS])
    detector = DeficiencyDetector(
        n_estimators=cfg["deficiency_rf"]["n_estimators"],
        max_depth=cfg["deficiency_rf"]["max_depth"],
    )

    X_list, y_list = [], []
    for _, group in df.groupby("user_id"):
        values = group[NUTRIENT_COLS].values
        for i in range(window, len(values)):
            window_data = values[i - window: i]
            mean_pct = window_data.mean(axis=0) / (rda + 1e-8)
            label = (mean_pct < detector.DEFICIENCY_THRESHOLD).astype(int)
            X_list.append(mean_pct)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    detector.fit(X_train, y_train)
    preds = detector.predict(X_val)
    accuracy = (preds == y_val).mean()
    print(f"  DeficiencyDetector — Val Label Accuracy: {accuracy:.4f}")

    save_path = cfg["deficiency_rf"]["save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    detector.save(save_path)
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Train LSTM Trend Predictor
# ---------------------------------------------------------------------------

def train_lstm(df: pd.DataFrame, cfg: dict, lstm_cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_series = []
    for _, group in df.groupby("user_id"):
        if len(group) > lstm_cfg["seq_len"] + 1:
            all_series.append(group[NUTRIENT_COLS])

    full_df = pd.concat(all_series, ignore_index=True)
    dataset = DailyIntakeDataset(full_df, seq_len=lstm_cfg["seq_len"], nutrient_cols=NUTRIENT_COLS)

    val_size = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    model = NutritionTrendLSTM(
        input_size=len(NUTRIENT_COLS),
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lstm_cfg["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    save_path = lstm_cfg["save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, lstm_cfg["epochs"] + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    val_loss += criterion(model(X), y).item() * len(y)
            val_loss /= len(val_loader.dataset)
            print(f"  LSTM Epoch {epoch:04d} | Val MSE: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict()}, save_path)

    print(f"  LSTM saved -> {save_path}")


# ---------------------------------------------------------------------------
# Train Anomaly Detector
# ---------------------------------------------------------------------------

def train_anomaly_detector(df: pd.DataFrame, cfg: dict):
    X = df[NUTRIENT_COLS].values
    detector = AnomalyDetector(
        contamination=cfg["contamination"],
        n_estimators=cfg["n_estimators"],
    )
    detector.fit(X)
    labels = detector.predict(X)
    anomaly_pct = (labels == -1).mean() * 100
    print(f"  AnomalyDetector — flagged {anomaly_pct:.1f}% of samples as anomalies")
    save_path = cfg["save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    detector.save(save_path)
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    health_cfg = cfg["health_model"]

    log_path = "data/processed/daily_logs.csv"
    if Path(log_path).exists():
        df = pd.read_csv(log_path)
        print(f"Loaded {len(df)} real daily logs")
    else:
        print("No real logs found — generating synthetic training data")
        df = generate_synthetic_logs(n_users=200, days=120)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_csv(log_path, index=False)
        print(f"Synthetic logs saved to {log_path}")

    print("\n[1/3] Training DeficiencyDetector...")
    train_deficiency_detector(df, health_cfg)

    print("\n[2/3] Training NutritionTrendLSTM...")
    train_lstm(df, health_cfg, health_cfg["lstm"])

    print("\n[3/3] Training AnomalyDetector...")
    train_anomaly_detector(df, health_cfg["anomaly"])

    print("\nAll health models trained successfully.")


if __name__ == "__main__":
    main()
