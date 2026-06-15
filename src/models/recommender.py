"""
Content-Based Food Recommender.

Uses cosine similarity between nutritional vectors to recommend foods
that fill the user's detected nutrient deficiencies.

No external APIs — runs entirely on the USDA nutrition CSV loaded locally.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib


NUTRITION_FEATURES = [
    "calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
    "sugar_g", "sodium_mg", "iron_mg", "calcium_mg", "vitamin_c_mg"
]

DAILY_RDA = {
    "calories": 2000, "protein_g": 50, "carbs_g": 275, "fat_g": 78,
    "fiber_g": 28, "sugar_g": 50, "sodium_mg": 2300, "iron_mg": 18,
    "calcium_mg": 1000, "vitamin_c_mg": 90,
}


class FoodRecommender:
    """
    Content-based recommender using cosine similarity on nutrition vectors.

    Workflow:
      1. fit()  — normalize USDA nutrition data, build food vectors
      2. recommend() — given deficient nutrients, find top-K foods that cover them
    """

    def __init__(self):
        self.food_names: list = []
        self.food_vectors: np.ndarray = None   # (N_foods, N_features) normalized
        self.scaler = MinMaxScaler()

    def fit(self, nutrition_df: pd.DataFrame):
        df = nutrition_df.copy()
        df = df.dropna(subset=NUTRITION_FEATURES)
        self.food_names = df["description"].tolist()
        raw = df[NUTRITION_FEATURES].values.astype(np.float32)
        self.food_vectors = self.scaler.fit_transform(raw)

    def recommend(self, deficient_nutrients: list, consumed_today: dict,
                  top_k: int = 10, exclude_foods: list = None) -> list:
        """
        deficient_nutrients: list of nutrient column names that are low
        consumed_today     : {nutrient: amount_consumed} for the day so far
        Returns: list of dicts with food name + nutritional info
        """
        if self.food_vectors is None:
            raise RuntimeError("Call fit() before recommend()")

        # Build a query vector: high weight on deficient nutrients, zero elsewhere
        query = np.zeros(len(NUTRITION_FEATURES), dtype=np.float32)
        for i, nutrient in enumerate(NUTRITION_FEATURES):
            if nutrient in deficient_nutrients:
                query[i] = 1.0

        if query.sum() == 0:
            # No deficiencies — recommend balanced foods by overall nutrition score
            query = np.ones(len(NUTRITION_FEATURES), dtype=np.float32)

        scores = cosine_similarity(query.reshape(1, -1), self.food_vectors)[0]

        if exclude_foods:
            exclude_set = set(f.lower() for f in exclude_foods)
            for i, name in enumerate(self.food_names):
                if name.lower() in exclude_set:
                    scores[i] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "food": self.food_names[idx],
                "score": float(scores[idx]),
                "nutrients": {
                    feat: float(self.food_vectors[idx, i] * (self.scaler.data_max_[i] - self.scaler.data_min_[i]) + self.scaler.data_min_[i])
                    for i, feat in enumerate(NUTRITION_FEATURES)
                }
            })
        return results

    def gap_based_recommend(self, daily_targets: dict, consumed_today: dict, top_k: int = 10) -> list:
        """
        Recommend foods that best fill today's remaining nutritional gap.
        consumed_today: {nutrient: amount_already_eaten}
        """
        gap = {}
        for nutrient in NUTRITION_FEATURES:
            target = daily_targets.get(nutrient, DAILY_RDA.get(nutrient, 1))
            consumed = consumed_today.get(nutrient, 0)
            gap[nutrient] = max(0, target - consumed)

        # Normalize gap vector
        gap_vec = np.array([gap.get(n, 0) for n in NUTRITION_FEATURES], dtype=np.float32)
        rda_vec = np.array([DAILY_RDA.get(n, 1) for n in NUTRITION_FEATURES], dtype=np.float32)
        query = gap_vec / (rda_vec + 1e-8)

        scores = cosine_similarity(query.reshape(1, -1), self.food_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "food": self.food_names[idx],
                "match_score": float(scores[idx]),
            })
        return results

    def save(self, path: str):
        joblib.dump({
            "food_names": self.food_names,
            "food_vectors": self.food_vectors,
            "scaler": self.scaler,
        }, path)

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.food_names = data["food_names"]
        obj.food_vectors = data["food_vectors"]
        obj.scaler = data["scaler"]
        return obj
