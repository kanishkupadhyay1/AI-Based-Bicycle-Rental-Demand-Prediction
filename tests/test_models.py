"""Tests for model artifacts, loading, and prediction behavior."""

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    GAM_MODEL_PATH,
    GB_MODEL_PATH,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    TEST_CSV,
)
from src.features.feature_engineering import get_raw_features_and_target


def test_all_model_artifacts_exist():
    for path in [GAM_MODEL_PATH, LR_MODEL_PATH, RF_MODEL_PATH, GB_MODEL_PATH, SCALER_PATH]:
        assert path.exists(), f"Missing model artifact: {path}"


def test_models_load_and_predict_real_test_rows():
    test_df = pd.read_csv(TEST_CSV).head(10)
    X_test_df, _ = get_raw_features_and_target(test_df)

    preprocessor = joblib.load(SCALER_PATH)
    X_transformed = preprocessor.transform(X_test_df)
    X_gam = X_test_df[ALL_FEATURES].to_numpy(dtype=float)

    gam = joblib.load(GAM_MODEL_PATH)
    lr = joblib.load(LR_MODEL_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    gb = joblib.load(GB_MODEL_PATH)

    gam_preds = gam.predict(X_gam)
    lr_preds = np.clip(lr.predict(X_transformed), 0.0, None)
    rf_preds = np.clip(rf.predict(X_transformed), 0.0, None)
    gb_preds = np.clip(gb.predict(X_transformed), 0.0, None)

    for preds in [gam_preds, lr_preds, rf_preds, gb_preds]:
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0)
