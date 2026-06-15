"""
Fine-tune EfficientNet-B3 on Food-101 + Indian Food (181 classes).

Strategy:
  Phase 1 — freeze backbone, train new head only (5 epochs, fast)
  Phase 2 — unfreeze all, fine-tune with low LR (up to 20 epochs, AMP)

Starts from the existing Food-101 checkpoint (101 classes).

Usage:
  python -m src.training.train_combined_classifier
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np

from src.data.combined_dataset import CombinedFoodDataset
from src.data.preprocessing import get_train_transforms, get_val_transforms, AlbumentationsWrapper
from src.models.food_classifier import FoodClassifier
from src.utils.metrics import compute_topk_accuracy


# ── helpers ──────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: CombinedFoodDataset) -> WeightedRandomSampler:
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    n_classes = dataset.num_classes
    class_weight = {cls: 1.0 / counts[cls] for cls in counts}
    sample_weights = [class_weight[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    cc_cfg  = cfg["combined_classifier"]
    fc_cfg  = cfg["food_classifier"]
    data_cfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_transform = AlbumentationsWrapper(get_train_transforms(data_cfg["image_size"]))
    val_transform   = AlbumentationsWrapper(get_val_transforms(data_cfg["image_size"]))

    train_ds = CombinedFoodDataset(data_cfg["food101_dir"], split="train", transform=train_transform)
    val_ds   = CombinedFoodDataset(data_cfg["food101_dir"], split="test",  transform=val_transform)

    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    print(f"Total classes: {train_ds.num_classes}  (Food-101: 101 + Indian: {train_ds.num_classes - 101})")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=cc_cfg["batch_size"],
                              sampler=sampler, num_workers=data_cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cc_cfg["batch_size"],
                              shuffle=False, num_workers=data_cfg["num_workers"], pin_memory=True)

    # Build model with new num_classes
    model = FoodClassifier(
        num_classes=train_ds.num_classes,
        model_name=fc_cfg["model_name"],
        pretrained=False,
        dropout=fc_cfg["dropout"],
    ).to(device)

    # Load Food-101 checkpoint backbone weights (classifier layer excluded)
    food101_ckpt = Path(fc_cfg["save_path"])
    if food101_ckpt.exists():
        print(f"Loading backbone from Food-101 checkpoint: {food101_ckpt}")
        ckpt = torch.load(food101_ckpt, map_location=device)
        state = ckpt["model_state_dict"]
        # Drop old classifier weights (size mismatch) — only load backbone
        filtered = {k: v for k, v in state.items() if not k.startswith("classifier")}
        model.load_state_dict(filtered, strict=False)
        print("  Backbone loaded. Classifier head re-initialised for 181 classes.")
    else:
        print("No Food-101 checkpoint found — training from ImageNet pretrained weights.")
        model = FoodClassifier(
            num_classes=train_ds.num_classes,
            model_name=fc_cfg["model_name"],
            pretrained=True,
            dropout=fc_cfg["dropout"],
        ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler("cuda")

    save_path = Path(cc_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_top1 = 0.0

    # ── Resume from combined checkpoint if it exists ──────────────────────────
    resume_phase2 = False
    if save_path.exists():
        print(f"Found existing combined checkpoint: {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_top1 = ckpt.get("val_top1", 0.0)
        epoch_tag  = ckpt.get("epoch", "")
        print(f"  Loaded checkpoint — epoch: {epoch_tag}, Top-1: {best_top1:.4f}")
        if str(epoch_tag).startswith("p2") or str(epoch_tag).startswith("P2"):
            resume_phase2 = True
            print("  Resuming Phase 2 fine-tuning (skipping Phase 1).")

    if not resume_phase2:
        # ── Phase 1: freeze backbone, train head only ─────────────────────────
        print("\n=== Phase 1: Train head only (backbone frozen) ===")
        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.classifier.parameters(), lr=cc_cfg["phase1_lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cc_cfg["phase1_epochs"])

        for epoch in range(1, cc_cfg["phase1_epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"P1 Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Top1: {val_top1:.4f} Top5: {val_top5:.4f}")
            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save({
                    "epoch": f"p1_{epoch}",
                    "model_state_dict": model.state_dict(),
                    "val_top1": val_top1,
                    "class_to_idx": train_ds.class_to_idx,
                    "idx_to_class": train_ds.idx_to_class,
                    "num_classes": train_ds.num_classes,
                }, save_path)
                print(f"  --> Saved (Top-1: {val_top1:.4f})")

    # ── Phase 2: unfreeze all, fine-tune ─────────────────────────────────────
    print("\n=== Phase 2: Fine-tune all layers ===")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=cc_cfg["phase2_lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cc_cfg["phase2_epochs"])
    patience_counter = 0

    for epoch in range(1, cc_cfg["phase2_epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"P2 Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Top1: {val_top1:.4f} Top5: {val_top5:.4f}")
        if val_top1 > best_top1:
            best_top1 = val_top1
            patience_counter = 0
            torch.save({
                "epoch": f"p2_{epoch}",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "class_to_idx": train_ds.class_to_idx,
                "idx_to_class": train_ds.idx_to_class,
                "num_classes": train_ds.num_classes,
            }, save_path)
            print(f"  --> Saved best (Top-1: {val_top1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cc_cfg["patience"]:
                print(f"Early stopping at phase-2 epoch {epoch}")
                break

    print(f"\nDone. Best Top-1: {best_top1:.4f}  ->  {save_path}")


if __name__ == "__main__":
    main()
