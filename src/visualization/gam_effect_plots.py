"""Generate GAM partial dependence effect curves and structured effect data."""

from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import GAM_MODEL_PATH, PLOTS_DIR, ensure_directories, get_logger
from src.models.train_gam import GAM_SMOOTH_TERMS

logger = get_logger(__name__)


def extract_gam_partial_effects(model_path: Path = GAM_MODEL_PATH) -> list[dict[str, Any]]:
    """Extract zero-centered partial dependence curves and 95% CIs for all GAM smooth terms."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained GAM model not found at {model_path}")

    gam = joblib.load(model_path)
    effects: list[dict[str, Any]] = []

    for term_info in GAM_SMOOTH_TERMS:
        term_idx = term_info["term_idx"]
        feature = term_info["feature"]
        label = term_info["label"]

        XX = gam.generate_X_grid(term=term_idx)
        # Ensure all other features in XX lie within their valid domain
        baseline_values = [1.0, 1.0, 6.0, 12.0, 0.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.6, 0.15]
        for col_i, val in enumerate(baseline_values):
            if col_i != term_idx:
                XX[:, col_i] = val

        pdep, confi = gam.partial_dependence(term=term_idx, X=XX, width=0.95)
        x_vals = XX[:, term_idx]

        # Center partial effect around zero for interpretability
        mean_effect = float(np.mean(pdep))
        pdep_centered = pdep - mean_effect
        ci_lower = confi[:, 0] - mean_effect
        ci_upper = confi[:, 1] - mean_effect

        points = [
            {
                "x": round(float(x), 4),
                "effect": round(float(y), 4),
                "ci_lower": round(float(lo), 4),
                "ci_upper": round(float(hi), 4),
            }
            for x, y, lo, hi in zip(x_vals, pdep_centered, ci_lower, ci_upper)
        ]

        effects.append(
            {
                "feature": feature,
                "label": label,
                "term_idx": term_idx,
                "points": points,
            }
        )

    return effects


def generate_gam_partial_effect_plots(
    model_path: Path = GAM_MODEL_PATH,
    plots_dir: Path = PLOTS_DIR,
) -> Path:
    """Plot and save GAM smooth partial dependence curves with 95% confidence bands."""
    ensure_directories()
    effects = extract_gam_partial_effects(model_path=model_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for idx, eff in enumerate(effects):
        ax = axes_flat[idx]
        x = [p["x"] for p in eff["points"]]
        y = [p["effect"] for p in eff["points"]]
        lo = [p["ci_lower"] for p in eff["points"]]
        hi = [p["ci_upper"] for p in eff["points"]]

        ax.plot(x, y, color="#1d4ed8", linewidth=2.2, label="Partial Effect")
        ax.fill_between(x, lo, hi, color="#93c5fd", alpha=0.35, label="95% CI")
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(eff["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel(eff["feature"])
        ax.set_ylabel("Centered Partial Effect (log-link)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    plt.suptitle(
        "PoissonGAM Smooth Partial Dependence Effects (95% Confidence Intervals)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = plots_dir / "gam_partial_effects.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved GAM partial effects plot to %s", output_path)
    return output_path


if __name__ == "__main__":
    generate_gam_partial_effect_plots()
