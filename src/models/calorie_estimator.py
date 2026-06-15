"""
Calorie & Macro Estimator — Multi-output MLP regression.

Input:  101-dim softmax vector from FoodClassifier (class probabilities)
Output: [calories, protein_g, carbs_g, fat_g, fiber_g]

Trained on USDA FoodData Central nutrition data aligned to Food-101 classes.
"""

import torch
import torch.nn as nn


class CalorieEstimator(nn.Module):
    def __init__(self, input_dim: int = 101, hidden_dims: list = None, output_dim: int = 5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, class_probs: torch.Tensor) -> torch.Tensor:
        return self.net(class_probs)


class CalorieLoss(nn.Module):
    """Weighted MSE — penalizes calorie errors more than micronutrient errors."""
    def __init__(self, weights: list = None):
        super().__init__()
        # weights: [calories, protein, carbs, fat, fiber]
        self.weights = torch.tensor(weights or [2.0, 1.0, 1.0, 1.0, 0.5])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(pred.device)
        return ((pred - target) ** 2 * w).mean()


def load_model(checkpoint_path: str, input_dim: int = 101, device: str = "cpu") -> CalorieEstimator:
    model = CalorieEstimator(input_dim=input_dim)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model
