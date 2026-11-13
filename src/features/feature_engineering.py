"""Feature engineering and preprocessing transformations for ML models."""

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_CONFIG_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TEMPORAL_CONTINUOUS_FEATURES,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

NUMERIC_SCALED_FEATURES = TEMPORAL_CONTINUOUS_FEATURES + CONTINUOUS_FEATURES


def get_raw_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract ordered feature DataFrame and target array from a processed DataFrame."""
    missing = [col for col in ALL_FEATURES + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer for scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_SCALED_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_feature_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Compute real feature metadata, bounds, and defaults from the dataset."""
    metadata: dict[str, Any] = {
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "temporal_continuous_features": TEMPORAL_CONTINUOUS_FEATURES,
        "target": TARGET_COLUMN,
        "ranges": {},
    }

    for col in ALL_FEATURES:
        series = df[col]
        is_cat = col in CATEGORICAL_FEATURES or col in TEMPORAL_CONTINUOUS_FEATURES
        metadata["ranges"][col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "default": int(series.median()) if is_cat else round(float(series.median()), 4),
            "unique_values": sorted(int(v) for v in series.unique()) if col in CATEGORICAL_FEATURES else None,
        }
    return metadata


def fit_and_save_feature_pipeline(
    train_df: pd.DataFrame,
    scaler_path: Path = SCALER_PATH,
    config_path: Path = FEATURE_CONFIG_PATH,
) -> Tuple[ColumnTransformer, dict[str, Any]]:
    """Fit preprocessor on training data only and persist preprocessor + feature config."""
    ensure_directories()
    X_train, _ = get_raw_features_and_target(train_df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, scaler_path)
    logger.info("Saved fitted feature preprocessor to %s", scaler_path)

    feature_config = build_feature_metadata(train_df)
    feature_config["transformed_feature_names"] = list(preprocessor.get_feature_names_out())
    config_path.write_text(json.dumps(feature_config, indent=2), encoding="utf-8")
    logger.info("Saved feature configuration to %s", config_path)

    return preprocessor, feature_config


def load_feature_preprocessor(scaler_path: Path = SCALER_PATH) -> ColumnTransformer:
    """Load the persisted scikit-learn feature preprocessor."""
    if not scaler_path.exists():
        raise FileNotFoundError(f"Feature preprocessor not found at {scaler_path}")
    return joblib.load(scaler_path)
