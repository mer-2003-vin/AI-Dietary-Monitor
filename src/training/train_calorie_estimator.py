"""
Train the CalorieEstimator (multi-output regression) on USDA nutrition data.

Usage:
  python -m src.training.train_calorie_estimator

Requires:
  - Trained food classifier checkpoint (to get class names)
  - data/raw/usda_nutrition.csv
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm

from src.data.nutrition_dataset import NutritionDataset
from src.models.calorie_estimator import CalorieEstimator, CalorieLoss
from src.utils.metrics import regression_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(y)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = regression_metrics(preds.numpy(), targets.numpy())
    return total_loss / len(loader.dataset), metrics


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ce_cfg   = cfg["calorie_estimator"]
    data_cfg = cfg["data"]
    fc_cfg   = cfg["food_classifier"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load class names from the food classifier checkpoint
    fc_checkpoint = torch.load(fc_cfg["save_path"], map_location="cpu")
    food101_classes = list(fc_checkpoint["class_to_idx"].keys())

    scaler_path = "saved_models/calorie_estimator/scaler.joblib"
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    dataset = NutritionDataset(
        csv_path=data_cfg["usda_nutrition_csv"],
        food101_classes=food101_classes,
        scaler_path=scaler_path,
        fit_scaler=True,
    )

    val_size  = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=ce_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=ce_cfg["batch_size"], shuffle=False)

    model = CalorieEstimator(
        input_dim=len(food101_classes),
        hidden_dims=ce_cfg["hidden_dims"],
        output_dim=len(ce_cfg["output_targets"]),
    ).to(device)

    criterion = CalorieLoss()
    optimizer = AdamW(model.parameters(), lr=ce_cfg["lr"])

    save_path = Path(ce_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, ce_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "target_names": ce_cfg["output_targets"],
            }, save_path)

    print(f"Training complete. Best Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
