"""Evaluate all trained models using MAE, RMSE, and Poisson Deviance Explained."""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    ALL_FEATURES,
    DATE_COLUMN,
    GAM_MODEL_PATH,
    GB_MODEL_PATH,
    LR_MODEL_PATH,
    MODEL_COMPARISON_CSV,
    RF_MODEL_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_CSV,
    TEST_PREDICTIONS_CSV,
    TRAIN_CSV,
    ensure_directories,
    get_logger,
)
from src.features.feature_engineering import get_raw_features_and_target, load_feature_preprocessor

logger = get_logger(__name__)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Compute total Poisson deviance D(y, mu) = 2 * sum(y * log(y/mu) - (y - mu)).

    Predictions are floored at `eps` to prevent log(0) when evaluating models
    that can produce zero or near-zero predictions.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), eps, None)

    term = np.zeros_like(y_t, dtype=float)
    pos_mask = y_t > 0
    term[pos_mask] = y_t[pos_mask] * np.log(y_t[pos_mask] / y_p[pos_mask])
    deviance = 2.0 * np.sum(term - (y_t - y_p))
    return float(deviance)


def compute_poisson_deviance_explained(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_mean: float,
) -> float:
    """Compute Poisson deviance explained: 1 - D(y, y_pred) / D(y, null_mean)."""
    y_null = np.full_like(y_true, fill_value=float(null_mean), dtype=float)
    d_null = poisson_deviance(y_true, y_null)
    if d_null <= 0:
        return 0.0
    d_model = poisson_deviance(y_true, y_pred)
    return float(1.0 - (d_model / d_null))


def evaluate_all_models(
    train_path: Path = TRAIN_CSV,
    test_path: Path = TEST_CSV,
    metrics_path: Path = MODEL_COMPARISON_CSV,
    predictions_path: Path = TEST_PREDICTIONS_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load all trained models, evaluate on chronological test set, and save results."""
    ensure_directories()

    for required in [
        train_path,
        test_path,
        GAM_MODEL_PATH,
        LR_MODEL_PATH,
        RF_MODEL_PATH,
        GB_MODEL_PATH,
        SCALER_PATH,
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Required file missing for evaluation: {required}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    _, y_train = get_raw_features_and_target(train_df)
    X_test_df, y_test = get_raw_features_and_target(test_df)
    null_mean = float(np.mean(y_train))

    preprocessor = load_feature_preprocessor(SCALER_PATH)
    X_test_transformed = preprocessor.transform(X_test_df)
    X_test_gam = X_test_df[ALL_FEATURES].to_numpy(dtype=float)

    gam_model = joblib.load(GAM_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    gb_model = joblib.load(GB_MODEL_PATH)

    raw_predictions = {
        "Generalized Additive Model (GAM)": gam_model.predict(X_test_gam),
        "Linear Regression": lr_model.predict(X_test_transformed),
        "Random Forest Regressor": rf_model.predict(X_test_transformed),
        "Gradient Boosting Regressor": gb_model.predict(X_test_transformed),
    }

    # Ensure non-negative rental count predictions
    predictions = {name: np.clip(pred, 0.0, None) for name, pred in raw_predictions.items()}

    gam_native_dev = float(
        gam_model.statistics_.get("pseudo_r2", {}).get("explained_deviance", np.nan)
    )

    metrics_rows = []
    for model_name, y_pred in predictions.items():
        mae = compute_mae(y_test, y_pred)
        rmse = compute_rmse(y_test, y_pred)
        dev_explained = compute_poisson_deviance_explained(y_test, y_pred, null_mean=null_mean)
        metrics_rows.append(
            {
                "model": model_name,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "deviance_explained": round(dev_explained, 4),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved model comparison metrics to %s", metrics_path)

    pred_df = pd.DataFrame(
        {
            DATE_COLUMN: test_df[DATE_COLUMN],
            "hr": test_df["hr"],
            "actual_cnt": y_test.astype(int),
            "gam_pred": np.round(predictions["Generalized Additive Model (GAM)"], 2),
            "linear_regression_pred": np.round(predictions["Linear Regression"], 2),
            "random_forest_pred": np.round(predictions["Random Forest Regressor"], 2),
            "gradient_boosting_pred": np.round(predictions["Gradient Boosting Regressor"], 2),
        }
    )
    pred_df.to_csv(predictions_path, index=False)
    logger.info("Saved test predictions to %s", predictions_path)

    best_mae_row = metrics_df.loc[metrics_df["mae"].idxmin()]
    best_rmse_row = metrics_df.loc[metrics_df["rmse"].idxmin()]

    summary = {
        "best_model_by_mae": str(best_mae_row["model"]),
        "best_mae": float(best_mae_row["mae"]),
        "best_model_by_rmse": str(best_rmse_row["model"]),
        "best_rmse": float(best_rmse_row["rmse"]),
        "gam_train_native_deviance_explained": round(gam_native_dev, 4),
        "test_samples": len(test_df),
        "train_samples": len(train_df),
    }

    print("\n" + "=" * 78)
    print("MODEL PERFORMANCE COMPARISON (Chronological Test Set)")
    print("=" * 78)
    print(metrics_df.to_string(index=False))
    print("-" * 78)
    print(f"Best Model by MAE : {summary['best_model_by_mae']} (MAE = {summary['best_mae']:.4f})")
    print(f"Best Model by RMSE: {summary['best_model_by_rmse']} (RMSE = {summary['best_rmse']:.4f})")
    print(f"GAM Native Training Deviance Explained: {summary['gam_train_native_deviance_explained']:.4f}")
    print("=" * 78 + "\n")

    return metrics_df, pred_df, summary


if __name__ == "__main__":
    evaluate_all_models()
