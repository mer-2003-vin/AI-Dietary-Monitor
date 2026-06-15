"""
Train EfficientNet on Food-101 with:
  - Resume from checkpoint
  - Mixed precision (AMP) — 2x speedup
  - EfficientNet-B0 (3x faster than B3, ~80% Top-1)

Usage:
  python -m src.training.train_food_classifier
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

from src.data.food_dataset import Food101Dataset
from src.data.preprocessing import get_train_transforms, get_val_transforms, AlbumentationsWrapper
from src.models.food_classifier import FoodClassifier
from src.utils.metrics import compute_topk_accuracy


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, top1, top5, total = 0.0, 0, 0, 0
    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        t1, t5 = compute_topk_accuracy(logits, labels, topk=(1, 5))
        top1 += t1 * len(labels)
        top5 += t5 * len(labels)
        total += len(labels)
    return total_loss / total, top1 / total, top5 / total


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    fc_cfg   = cfg["food_classifier"]
    data_cfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_transform = AlbumentationsWrapper(get_train_transforms(data_cfg["image_size"]))
    val_transform   = AlbumentationsWrapper(get_val_transforms(data_cfg["image_size"]))

    train_ds = Food101Dataset(data_cfg["food101_dir"], split="train", transform=train_transform)
    val_ds   = Food101Dataset(data_cfg["food101_dir"], split="test",  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=fc_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=fc_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=True)

    model = FoodClassifier(
        num_classes=train_ds.num_classes,
        model_name=fc_cfg["model_name"],
        pretrained=fc_cfg["pretrained"],
        dropout=fc_cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=fc_cfg["lr"], weight_decay=fc_cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=fc_cfg["epochs"])
    scaler    = GradScaler("cuda")

    save_path = Path(fc_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint ---
    start_epoch   = 1
    best_top1     = 0.0
    patience_counter = 0

    if save_path.exists():
        print(f"Resuming from checkpoint: {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_top1   = ckpt.get("val_top1", 0.0)
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"  Resumed from epoch {start_epoch - 1} | Best Top-1: {best_top1:.4f}")
        # Fast-forward scheduler to match resumed epoch
        for _ in range(start_epoch - 1):
            scheduler.step()

    for epoch in range(start_epoch, fc_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Top1: {val_top1:.4f} Top5: {val_top5:.4f}")

        if val_top1 > best_top1:
            best_top1 = val_top1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "class_to_idx": train_ds.class_to_idx,
                "idx_to_class": train_ds.idx_to_class,
            }, save_path)
            print(f"  --> Saved best model (Top-1: {val_top1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= fc_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training complete. Best Top-1: {best_top1:.4f}")


if __name__ == "__main__":
    main()
