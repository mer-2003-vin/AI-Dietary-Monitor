"""
Health Analytics — three trained models:

1. DeficiencyDetector   : RandomForest classifier
   Input : daily % of RDA met per nutrient (10 features)
   Output: multi-label — which nutrients are deficient

2. NutritionTrendLSTM  : LSTM sequence model
   Input : 14-day window of daily nutrient intake (seq_len x 10)
   Output: predicted next-day intake (10 values)

3. AnomalyDetector     : Isolation Forest
   Input : daily nutrient intake vector (10 features)
   Output: anomaly score (-1 = anomaly, 1 = normal)
"""

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler


NUTRIENT_COLS = [
    "calories", "protein_g", "carbs_g", "fat_g",
    "fiber_g", "sugar_g", "sodium_mg", "iron_mg", "calcium_mg", "vitamin_c_mg"
]

DAILY_RDA = {
    "calories": 2000, "protein_g": 50, "carbs_g": 275, "fat_g": 78,
    "fiber_g": 28, "sugar_g": 50, "sodium_mg": 2300, "iron_mg": 18,
    "calcium_mg": 1000, "vitamin_c_mg": 90,
}


# ---------------------------------------------------------------------------
# 1. Deficiency Detector
# ---------------------------------------------------------------------------

class DeficiencyDetector:
    """
    Multi-label RandomForest: predicts which nutrients are below threshold.
    Features: % RDA met for each nutrient over past N days.
    Labels  : binary vector — 1 if nutrient < 70% RDA, 0 otherwise.
    """

    DEFICIENCY_THRESHOLD = 0.70  # below 70% RDA → deficient

    def __init__(self, n_estimators: int = 200, max_depth: int = 10):
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model = MultiOutputClassifier(base_rf)
        self.scaler = StandardScaler()
        self.nutrient_names = NUTRIENT_COLS

    def compute_features(self, daily_records: np.ndarray) -> np.ndarray:
        """daily_records: (N_days, 10) — raw intake values."""
        rda = np.array([DAILY_RDA[n] for n in NUTRIENT_COLS], dtype=np.float32)
        pct = daily_records / (rda + 1e-8)      # % RDA met each day
        return np.mean(pct, axis=0, keepdims=True)  # mean % RDA over window

    def compute_labels(self, mean_pct: np.ndarray) -> np.ndarray:
        return (mean_pct < self.DEFICIENCY_THRESHOLD).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        # proba is list of (n_samples, 2) arrays — extract positive class
        return np.column_stack([p[:, 1] for p in proba])

    def get_deficient_nutrients(self, daily_records: np.ndarray) -> list:
        features = self.compute_features(daily_records)
        pred = self.predict(features)[0]
        return [self.nutrient_names[i] for i, v in enumerate(pred) if v == 1]

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.nutrient_names = NUTRIENT_COLS
        return obj


# ---------------------------------------------------------------------------
# 2. Nutrition Trend LSTM
# ---------------------------------------------------------------------------

class NutritionTrendLSTM(nn.Module):
    """
    Predicts next-day nutrient intake from a 14-day window.
    Input : (batch, seq_len, input_size)
    Output: (batch, input_size)
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last timestep


def load_lstm(checkpoint_path: str, device: str = "cpu") -> NutritionTrendLSTM:
    model = NutritionTrendLSTM()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Isolation Forest on daily nutrient intake vectors."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns -1 for anomaly, 1 for normal."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score — more negative = more anomalous."""
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def is_anomaly(self, daily_intake: np.ndarray) -> bool:
        return self.predict(daily_intake.reshape(1, -1))[0] == -1

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        return obj
