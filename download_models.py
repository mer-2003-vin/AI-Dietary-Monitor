"""
Download model checkpoints from HF Hub at startup.
Runs automatically when the app starts on HF Spaces.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO = "shubeeeeee/ai-dietary-models"

MODELS = [
    ("food_classifier/best_model.pth",     "saved_models/food_classifier/best_model.pth"),
    ("combined_classifier/best_model.pth", "saved_models/combined_classifier/best_model.pth"),
    ("calorie_estimator/best_model.pth",   "saved_models/calorie_estimator/best_model.pth"),
    ("calorie_estimator/scaler.joblib",    "saved_models/calorie_estimator/scaler.joblib"),
    ("health_model/anomaly_detector.joblib","saved_models/health_model/anomaly_detector.joblib"),
    ("health_model/deficiency_rf.joblib",  "saved_models/health_model/deficiency_rf.joblib"),
    ("health_model/lstm_trend.pth",        "saved_models/health_model/lstm_trend.pth"),
    ("recommender/food_vectors.npz",       "saved_models/recommender/food_vectors.npz"),
]


def download_all():
    for repo_file, local_path in MODELS:
        dest = Path(local_path)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_file}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=repo_file,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves to ./ + filename; move to expected path
        downloaded = Path(repo_file)
        if downloaded.exists() and not dest.exists():
            downloaded.rename(dest)
        print(f"  Saved -> {local_path}", flush=True)


if __name__ == "__main__":
    download_all()
