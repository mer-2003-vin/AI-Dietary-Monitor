import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.item() / targets.size(0))
        return results


def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(targets, preds),
        "r2": r2_score(targets, preds),
        "rmse": float(np.sqrt(((preds - targets) ** 2).mean())),
    }
