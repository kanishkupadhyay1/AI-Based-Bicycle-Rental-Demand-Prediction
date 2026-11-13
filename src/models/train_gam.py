"""Train the primary Generalized Additive Model (PoissonGAM) using PyGAM."""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from pygam import PoissonGAM, f, s

from src.config import ALL_FEATURES, GAM_MODEL_PATH, TRAIN_CSV, ensure_directories, get_logger
from src.features.feature_engineering import get_raw_features_and_target

logger = get_logger(__name__)

# Feature indices in ALL_FEATURES:
# 0: season (f), 1: yr (f), 2: mnth (s), 3: hr (s), 4: holiday (f),
# 5: weekday (f), 6: workingday (f), 7: weathersit (f),
# 8: temp (s), 9: atemp (s), 10: hum (s), 11: windspeed (s)
GAM_SMOOTH_TERMS = [
    {"term_idx": 3, "feature": "hr", "label": "Hour of Day (hr)"},
    {"term_idx": 8, "feature": "temp", "label": "Normalized Temperature (temp)"},
    {"term_idx": 9, "feature": "atemp", "label": "Feeling Temperature (atemp)"},
    {"term_idx": 10, "feature": "hum", "label": "Normalized Humidity (hum)"},
    {"term_idx": 11, "feature": "windspeed", "label": "Normalized Windspeed (windspeed)"},
    {"term_idx": 2, "feature": "mnth", "label": "Month of Year (mnth)"},
]


def build_gam_formula():
    """Construct the PyGAM term specification for temporal and weather modeling."""
    return (
        f(0)
        + f(1)
        + s(2, n_splines=12, spline_order=3)
        + s(3, n_splines=24, spline_order=3)
        + f(4)
        + f(5)
        + f(6)
        + f(7)
        + s(8, n_splines=15, spline_order=3)
        + s(9, n_splines=15, spline_order=3)
        + s(10, n_splines=15, spline_order=3)
        + s(11, n_splines=12, spline_order=3)
    )


def train_gam_model(
    train_df: pd.DataFrame | None = None,
    model_path: Path = GAM_MODEL_PATH,
) -> Tuple[PoissonGAM, float]:
    """Fit PoissonGAM on training data with smoothing parameter selection.

    PoissonGAM with a log link is chosen because the target 'cnt' represents
    non-negative, right-skewed hourly bicycle rental counts.
    """
    ensure_directories()
    if train_df is None:
        if not TRAIN_CSV.exists():
            raise FileNotFoundError(f"Training data not found at {TRAIN_CSV}")
        train_df = pd.read_csv(TRAIN_CSV)

    X_train_df, y_train = get_raw_features_and_target(train_df)
    X_train = X_train_df[ALL_FEATURES].to_numpy(dtype=float)

    logger.info("Training PoissonGAM on %d training samples...", len(X_train))
    gam = PoissonGAM(build_gam_formula(), max_iter=200)

    # Select optimal smoothing parameter lambda on training data via GCV/UBRE
    lam_candidates = np.logspace(-2, 3, 6)
    gam.gridsearch(X_train, y_train, lam=lam_candidates, progress=False)

    pseudo_r2 = float(gam.statistics_.get("pseudo_r2", {}).get("explained_deviance", np.nan))
    logger.info(
        "PoissonGAM trained successfully. Training explained deviance (pseudo-R2): %.4f",
        pseudo_r2,
    )

    joblib.dump(gam, model_path)
    logger.info("Saved PoissonGAM model to %s", model_path)
    return gam, pseudo_r2


if __name__ == "__main__":
    train_gam_model()
