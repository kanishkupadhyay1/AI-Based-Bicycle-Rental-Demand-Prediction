"""Generate model evaluation and comparison plots from real test results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    MODEL_COMPARISON_CSV,
    PLOTS_DIR,
    TEST_PREDICTIONS_CSV,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def generate_evaluation_plots(
    metrics_path: Path = MODEL_COMPARISON_CSV,
    predictions_path: Path = TEST_PREDICTIONS_CSV,
    plots_dir: Path = PLOTS_DIR,
) -> list[Path]:
    """Generate model comparison, actual vs predicted, and residual plots."""
    ensure_directories()
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found at {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found at {predictions_path}")

    metrics_df = pd.read_csv(metrics_path)
    pred_df = pd.read_csv(predictions_path)
    saved_paths: list[Path] = []

    # 1. Model Comparison Chart (MAE, RMSE, Deviance Explained)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    short_names = [
        m.replace("Generalized Additive Model (GAM)", "GAM")
        .replace(" Regressor", "")
        .replace(" Regression", " Reg.")
        for m in metrics_df["model"]
    ]

    colors = ["#2563eb", "#64748b", "#059669", "#d97706"]
    axes[0].bar(short_names, metrics_df["mae"], color=colors)
    axes[0].set_title("Test MAE (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(short_names, metrics_df["rmse"], color=colors)
    axes[1].set_title("Test RMSE (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(short_names, metrics_df["deviance_explained"], color=colors)
    axes[2].set_title("Poisson Deviance Explained (Higher is Better)", fontweight="bold")
    axes[2].set_ylabel("Deviance Explained")
    axes[2].tick_params(axis="x", rotation=15)

    plt.suptitle("Model Performance Comparison on Chronological Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = plots_dir / "model_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Actual vs Predicted (2x2 scatter + sample time slice)
    model_cols = [
        ("gam_pred", "GAM"),
        ("linear_regression_pred", "Linear Regression"),
        ("random_forest_pred", "Random Forest"),
        ("gradient_boosting_pred", "Gradient Boosting"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    max_val = max(pred_df["actual_cnt"].max(), 900)
    for ax, (col, title) in zip(axes.flatten(), model_cols):
        ax.scatter(pred_df["actual_cnt"], pred_df[col], alpha=0.25, s=12, color="#2563eb")
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Ideal (y = x)")
        ax.set_title(f"{title}: Actual vs Predicted", fontweight="bold")
        ax.set_xlabel("Actual Count (cnt)")
        ax.set_ylabel("Predicted Count")
        ax.legend(loc="upper left", fontsize=8)
    plt.suptitle("Actual vs Predicted Hourly Bicycle Rentals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = plots_dir / "actual_vs_predicted.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Residual Plot (GAM & Best Tree Models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, title) in zip(axes.flatten(), model_cols):
        residuals = pred_df["actual_cnt"] - pred_df[col]
        ax.scatter(pred_df[col], residuals, alpha=0.25, s=12, color="#0284c7")
        ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.2)
        ax.set_title(f"{title}: Residuals vs Predicted", fontweight="bold")
        ax.set_xlabel("Predicted Count")
        ax.set_ylabel("Residual (Actual - Predicted)")
    plt.suptitle("Residual Analysis Across Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p3 = plots_dir / "residual_plot.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Prediction Error Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, title in model_cols:
        residuals = pred_df["actual_cnt"] - pred_df[col]
        sns.kdeplot(residuals, label=title, ax=ax, linewidth=2)
    ax.axvline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax.set_title("Prediction Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    p4 = plots_dir / "error_distribution.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(p4)

    logger.info("Generated %d evaluation plots in %s", len(saved_paths), plots_dir)
    return saved_paths


if __name__ == "__main__":
    generate_evaluation_plots()
