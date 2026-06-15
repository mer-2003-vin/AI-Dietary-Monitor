"""
End-to-end inference pipeline.

Flow:
  Image → FoodClassifier → class probs → CalorieEstimator → macros
                                        → class label (food name)
  History → DeficiencyDetector → deficient nutrients
           → AnomalyDetector   → anomaly flag
           → NutritionTrendLSTM → predicted tomorrow's intake
  Deficiencies → FoodRecommender → top-K food suggestions
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.data.preprocessing import get_val_transforms, AlbumentationsWrapper
from src.models.food_classifier import FoodClassifier, load_model as load_fc
from src.models.calorie_estimator import CalorieEstimator, load_model as load_ce
from src.models.health_analytics import (
    DeficiencyDetector, AnomalyDetector, load_lstm, NUTRIENT_COLS
)
from src.models.recommender import FoodRecommender
import joblib


class DietaryPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_models()

    def _load_models(self):
        fc_cfg = self.cfg["food_classifier"]
        ce_cfg = self.cfg["calorie_estimator"]
        hm_cfg = self.cfg["health_model"]
        rc_cfg = self.cfg["recommender"]

        # Food classifier — prefer combined (181-class) model if available
        cc_cfg = self.cfg.get("combined_classifier", {})
        combined_path = cc_cfg.get("save_path", "")
        classifier_path = (
            combined_path if combined_path and Path(combined_path).exists()
            else fc_cfg["save_path"]
        )
        self.food_classifier = load_fc(classifier_path, device=self.device)
        ckpt = torch.load(classifier_path, map_location="cpu")
        self.idx_to_class = ckpt["idx_to_class"]
        self.class_to_idx = ckpt["class_to_idx"]
        print(f"Loaded classifier: {classifier_path} ({len(self.idx_to_class)} classes)")

        # Calorie estimator
        self.calorie_estimator = load_ce(ce_cfg["save_path"], device=self.device)
        scaler_path = "saved_models/calorie_estimator/scaler.joblib"
        self.calorie_scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None

        # Health models
        self.deficiency_detector = DeficiencyDetector.load(hm_cfg["deficiency_rf"]["save_path"])
        self.anomaly_detector     = AnomalyDetector.load(hm_cfg["anomaly"]["save_path"])
        self.lstm_model           = load_lstm(hm_cfg["lstm"]["save_path"], device=self.device)

        # Recommender
        self.recommender = FoodRecommender.load(rc_cfg["save_path"])

        # Image transform
        self.transform = AlbumentationsWrapper(
            get_val_transforms(self.cfg["data"]["image_size"])
        )

    def _tta_transforms(self, image: Image.Image) -> torch.Tensor:
        """Return a batch of TTA-augmented tensors for a single image."""
        from src.data.preprocessing import AlbumentationsWrapper
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        _MEAN = [0.485, 0.456, 0.406]
        _STD  = [0.229, 0.224, 0.225]
        size = self.cfg["data"]["image_size"]

        variants = []
        for flip in [False, True]:
            for scale in [1.0, 0.9]:
                crop_size = int(size * scale)
                aug = A.Compose([
                    A.Resize(int(size * 1.14), int(size * 1.14)),
                    A.CenterCrop(crop_size, crop_size),
                    A.Resize(size, size),
                    A.HorizontalFlip(p=1.0 if flip else 0.0),
                    A.Normalize(mean=_MEAN, std=_STD),
                    ToTensorV2(),
                ])
                import numpy as np
                result = aug(image=np.array(image))["image"]
                variants.append(result)

        return torch.stack(variants).to(self.device)  # (N, C, H, W)

    def predict_food(self, image_path: str) -> dict:
        """Identify food from image and estimate nutrition."""
        image = Image.open(image_path).convert("RGB")

        # TTA: average logits over 4 augmented views
        batch = self._tta_transforms(image)
        with torch.no_grad():
            logits_mean = self.food_classifier(batch).mean(dim=0, keepdim=True)

        # Temperature scaling (T=0.5) sharpens confidence scores
        temperature = 0.5
        probs = torch.softmax(logits_mean / temperature, dim=1)

        top5_probs, top5_idx = probs.topk(5, dim=1)
        top5 = [
            {"food": self.idx_to_class[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top5_probs[0], top5_idx[0])
        ]

        # Calorie estimator was trained on 101-class probs — use top-101 slice or lookup by food
        n_ce = self.cfg["calorie_estimator"]["input_features"]
        probs_ce = probs[:, :n_ce] if probs.shape[1] >= n_ce else torch.nn.functional.pad(probs, (0, n_ce - probs.shape[1]))
        with torch.no_grad():
            macros_raw = self.calorie_estimator(probs_ce)

        macros = macros_raw.cpu().numpy()[0]
        if self.calorie_scaler:
            macros = self.calorie_scaler.inverse_transform(macros.reshape(1, -1))[0]

        target_names = self.cfg["calorie_estimator"]["output_targets"]
        nutrition = {name: max(0.0, float(val)) for name, val in zip(target_names, macros)}

        return {
            "top_predictions": top5,
            "predicted_food":  top5[0]["food"],
            "confidence":      top5[0]["confidence"],
            "nutrition":       nutrition,
        }

    def analyze_health(self, daily_records: np.ndarray) -> dict:
        """
        daily_records: np.ndarray of shape (N_days, 10) — past intake history.
        Columns must match NUTRIENT_COLS order.
        """
        deficient = self.deficiency_detector.get_deficient_nutrients(daily_records)
        is_anomaly = self.anomaly_detector.is_anomaly(daily_records[-1])

        # LSTM: predict tomorrow's intake
        from src.data.nutrition_dataset import DailyIntakeDataset
        seq_len = self.cfg["health_model"]["lstm"]["seq_len"]
        window = daily_records[-seq_len:] if len(daily_records) >= seq_len else daily_records
        # Simple normalize using mean/std of the window
        mean, std = window.mean(axis=0), window.std(axis=0) + 1e-8
        window_norm = (window - mean) / std
        tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.lstm_model(tensor).cpu().numpy()[0]
        predicted_tomorrow = (pred_norm * std + mean).tolist()

        return {
            "deficient_nutrients": deficient,
            "is_anomaly": bool(is_anomaly),
            "predicted_tomorrow": dict(zip(NUTRIENT_COLS, predicted_tomorrow)),
        }

    def get_recommendations(self, deficient_nutrients: list,
                            consumed_today: dict, top_k: int = 10) -> list:
        return self.recommender.recommend(
            deficient_nutrients=deficient_nutrients,
            consumed_today=consumed_today,
            top_k=top_k,
        )

    def full_analysis(self, image_path: str, daily_records: np.ndarray,
                      consumed_today: dict) -> dict:
        food_result   = self.predict_food(image_path)
        health_result = self.analyze_health(daily_records)
        recommendations = self.get_recommendations(
            deficient_nutrients=health_result["deficient_nutrients"],
            consumed_today=consumed_today,
        )
        return {
            "food":            food_result,
            "health":          health_result,
            "recommendations": recommendations,
        }
