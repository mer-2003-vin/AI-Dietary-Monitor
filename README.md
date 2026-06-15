---
title: AI Dietary Monitor
emoji: 🥗
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# AI-Powered Dietary Monitoring & Health Analytics System

An end-to-end ML system that recognizes food from photos, estimates nutrition, detects deficiencies, forecasts trends, and recommends meals — **no LLM APIs, all models trained from scratch**.

---

## Models

| Model | Architecture | Task | Accuracy |
|---|---|---|---|
| Food Classifier | EfficientNet-B3 | 181-class food recognition | 83.5% Top-1 |
| Calorie Estimator | MLP (256→128→64) | Macro prediction | — |
| Deficiency Detector | RandomForest | Nutrient deficiency detection | — |
| Trend Predictor | LSTM (14-day window) | Next-day intake forecast | — |
| Anomaly Detector | Isolation Forest | Unusual pattern detection | — |
| Food Recommender | Cosine similarity | Nutrition-based suggestions | — |

**181 classes** = Food-101 (101 international foods) + Indian Food Images dataset (80 Indian foods)  
Indian foods include: Aloo Tikki, Biryani, Butter Chicken, Dosa, Idli, Paneer Butter Masala, Samosa, Gulab Jamun, and 72 more.

---

## Setup

```bash
pip install -r requirements.txt
```

### Download datasets
- **Food-101**: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ → extract to `data/raw/food-101/`
- **Indian Food Images**: `kaggle datasets download -d iamsouravbanerjee/indian-food-images-dataset -p data/raw/indian-food --unzip`

### Download pre-trained model checkpoints
The large classifier checkpoints (>100MB) are hosted separately:
- `saved_models/food_classifier/best_model.pth` — EfficientNet-B3 Food-101 (84.3% Top-1)
- `saved_models/combined_classifier/best_model.pth` — EfficientNet-B3 Combined 181-class (83.5% Top-1)

> Share these via Google Drive with your partner.

---

## Train from scratch

```bash
# 1. Train food classifier (Food-101, 101 classes)
python -m src.training.train_food_classifier

# 2. Prepare nutrition data
python -m src.training.prepare_nutrition_data

# 3. Train calorie estimator
python -m src.training.train_calorie_estimator

# 4. Train health models (RandomForest + LSTM + IsolationForest)
python -m src.training.train_health_model

# 5. Train combined classifier (181 classes, requires step 1 checkpoint)
python -m src.training.train_combined_classifier
```

---

## Run the dashboard

```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501**

---

## Project Structure

```
├── configs/config.yaml          # Central config for all models
├── frontend/app.py              # Streamlit dashboard (glassmorphism dark theme)
├── src/
│   ├── data/                    # Dataset loaders & preprocessing
│   ├── models/                  # Model architectures
│   ├── training/                # Training scripts
│   └── inference/pipeline.py   # End-to-end inference pipeline
├── api/                         # FastAPI backend
├── saved_models/                # Trained checkpoints (small ones included)
└── requirements.txt
```

---

## Tech Stack

- **PyTorch** + **timm** (EfficientNet-B3)
- **scikit-learn** (RandomForest, IsolationForest)
- **Streamlit** (dashboard UI)
- **FastAPI** (REST API)
- **Albumentations** (image augmentation)
- **Plotly** (interactive charts)
