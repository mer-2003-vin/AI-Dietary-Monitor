"""
Build and save the FoodRecommender from USDA nutrition CSV.

Usage:
  python -m src.training.build_recommender
"""

import yaml
import pandas as pd
from pathlib import Path
from src.models.recommender import FoodRecommender


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    csv_path  = cfg["data"]["usda_nutrition_csv"]
    save_path = cfg["recommender"]["save_path"]

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    features = cfg["recommender"]["nutrition_features"] + ["description"]
    df = df.dropna(subset=cfg["recommender"]["nutrition_features"])

    recommender = FoodRecommender()
    recommender.fit(df)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    recommender.save(save_path)
    print(f"Recommender built with {len(recommender.food_names)} foods → {save_path}")


if __name__ == "__main__":
    main()
