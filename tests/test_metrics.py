"""Tests for evaluation metrics (MAE, RMSE, Poisson Deviance Explained)."""

import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_COMPARISON_CSV
from src.models.model_evaluation import (
    compute_mae,
    compute_poisson_deviance_explained,
    compute_rmse,
    poisson_deviance,
)


def test_mae_and_rmse_exact_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # errors: 2, 2, 3 -> MAE = 7/3, RMSE = sqrt((4+4+9)/3) = sqrt(17/3)
    assert compute_mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)
    assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt(17.0 / 3.0))


def test_poisson_deviance_perfect_prediction_is_zero():
    y_true = np.array([5.0, 10.0, 25.0])
    assert poisson_deviance(y_true, y_true) == pytest.approx(0.0, abs=1e-9)


def test_poisson_deviance_explained_bounds():
    y_true = np.array([10.0, 20.0, 40.0, 80.0])
    null_mean = float(np.mean(y_true))
    assert compute_poisson_deviance_explained(y_true, y_true, null_mean) == pytest.approx(1.0)
    y_null = np.full_like(y_true, null_mean)
    assert compute_poisson_deviance_explained(y_true, y_null, null_mean) == pytest.approx(0.0)


def test_model_comparison_csv_has_four_models():
    assert MODEL_COMPARISON_CSV.exists()
    df = pd.read_csv(MODEL_COMPARISON_CSV)
    assert len(df) == 4
    assert set(["model", "mae", "rmse", "deviance_explained"]).issubset(df.columns)
    assert (df["mae"] > 0).all()
    assert (df["rmse"] >= df["mae"]).all()
