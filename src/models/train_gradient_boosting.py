"""Train the Gradient Boosting Regressor comparison model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.config import GB_MODEL_PATH, SCALER_PATH, TRAIN_CSV, ensure_directories, get_logger
from src.features.feature_engineering import (
    fit_and_save_feature_pipeline,
    get_raw_features_and_target,
    load_feature_preprocessor,
)

logger = get_logger(__name__)


def train_gradient_boosting_model(
    train_df: pd.DataFrame | None = None,
    model_path: Path = GB_MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> GradientBoostingRegressor:
    """Fit GradientBoostingRegressor on encoded training features."""
    ensure_directories()
    if train_df is None:
        if not TRAIN_CSV.exists():
            raise FileNotFoundError(f"Training data not found at {TRAIN_CSV}")
        train_df = pd.read_csv(TRAIN_CSV)

    if not scaler_path.exists():
        preprocessor, _ = fit_and_save_feature_pipeline(train_df, scaler_path=scaler_path)
    else:
        preprocessor = load_feature_preprocessor(scaler_path)

    X_train_df, y_train = get_raw_features_and_target(train_df)
    X_train_transformed = preprocessor.transform(X_train_df)

    logger.info("Training GradientBoostingRegressor on %d training samples...", len(X_train_transformed))
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train_transformed, y_train)

    joblib.dump(model, model_path)
    logger.info("Saved GradientBoostingRegressor model to %s", model_path)
    return model


if __name__ == "__main__":
    train_gradient_boosting_model()
