"""Plotting utilities for nutrition dashboards and training curves."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List


NUTRIENT_COLS = [
    "calories", "protein_g", "carbs_g", "fat_g",
    "fiber_g", "sugar_g", "sodium_mg", "iron_mg", "calcium_mg", "vitamin_c_mg"
]

DAILY_RDA = {
    "calories": 2000, "protein_g": 50, "carbs_g": 275, "fat_g": 78,
    "fiber_g": 28, "sugar_g": 50, "sodium_mg": 2300, "iron_mg": 18,
    "calcium_mg": 1000, "vitamin_c_mg": 90,
}


def plot_daily_intake_vs_rda(intake: dict, save_path: str = None):
    nutrients = list(DAILY_RDA.keys())
    rda_vals  = [DAILY_RDA[n] for n in nutrients]
    consumed  = [intake.get(n, 0) for n in nutrients]
    pct       = [min(c / r * 100, 150) for c, r in zip(consumed, rda_vals)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if p < 70 else "#2ecc71" if p <= 100 else "#f39c12" for p in pct]
    bars = ax.barh(nutrients, pct, color=colors)
    ax.axvline(100, color="black", linestyle="--", linewidth=1.5, label="100% RDA")
    ax.axvline(70, color="orange", linestyle=":", linewidth=1, label="70% (deficiency threshold)")
    ax.set_xlabel("% of Daily RDA")
    ax.set_title("Today's Nutrient Intake vs. RDA")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_weekly_trend(df: pd.DataFrame, nutrient: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df[nutrient], marker="o", label="Actual", color="#3498db")
    ax.axhline(DAILY_RDA.get(nutrient, 0), color="red", linestyle="--", label="RDA")
    ax.set_title(f"{nutrient} — Weekly Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel(nutrient)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_macro_pie(calories: float, protein_g: float, carbs_g: float, fat_g: float):
    protein_cal = protein_g * 4
    carbs_cal   = carbs_g * 4
    fat_cal     = fat_g * 9
    labels = ["Protein", "Carbs", "Fat"]
    sizes  = [protein_cal, carbs_cal, fat_cal]
    colors = ["#3498db", "#e67e22", "#e74c3c"]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Macronutrient Distribution")
    return fig


def plot_training_curve(train_losses: List[float], val_losses: List[float],
                        metric_name: str = "Loss", save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label=f"Train {metric_name}")
    ax.plot(val_losses,   label=f"Val {metric_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Training Curve — {metric_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
