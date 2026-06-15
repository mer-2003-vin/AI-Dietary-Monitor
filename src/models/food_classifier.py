"""
Food Recognition Model — EfficientNet-B3 fine-tuned on Food-101.

Architecture:
  - Backbone: EfficientNet-B3 (pretrained on ImageNet via timm)
  - Head:     Dropout → Linear(1536 → 101)
  - Output:   101-class softmax (one per Food-101 category)
"""

import torch
import torch.nn as nn
import timm


class FoodClassifier(nn.Module):
    def __init__(self, num_classes: int = 101, model_name: str = "efficientnet_b3",
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings before the classification head."""
        return self.backbone(x)


def load_model(checkpoint_path: str, num_classes: int = None, device: str = "cpu") -> FoodClassifier:
    state = torch.load(checkpoint_path, map_location=device)
    # Infer num_classes from checkpoint if not provided
    if num_classes is None:
        num_classes = state.get("num_classes", len(state.get("idx_to_class", {})) or 101)
    model = FoodClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model
