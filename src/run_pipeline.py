"""End-to-end ML pipeline runner: download, validate, preprocess, train, evaluate, plot."""

import argparse

from src.config import (
    CLEANED_HOUR_CSV,
    GAM_MODEL_PATH,
    GB_MODEL_PATH,
    LR_MODEL_PATH,
    MODEL_COMPARISON_CSV,
    RF_MODEL_PATH,
    SCALER_PATH,
    TEST_CSV,
    TRAIN_CSV,
    get_logger,
)
from src.data.download_dataset import download_uci_bike_sharing_dataset
from src.data.preprocess import run_preprocessing
from src.data.validate_dataset import validate_dataset
from src.features.feature_engineering import fit_and_save_feature_pipeline
from src.models.model_evaluation import evaluate_all_models
from src.models.train_gam import train_gam_model
from src.models.train_gradient_boosting import train_gradient_boosting_model
from src.models.train_linear_regression import train_linear_regression_model
from src.models.train_random_forest import train_random_forest_model
from src.visualization.eda_visualizations import generate_eda_plots
from src.visualization.evaluation_plots import generate_evaluation_plots
from src.visualization.gam_effect_plots import generate_gam_partial_effect_plots

logger = get_logger(__name__)


def run_full_pipeline(force_retrain: bool = False) -> None:
    """Execute all ML pipeline stages in strict order."""
    logger.info("=== PHASE 2: Downloading Official UCI Bike Sharing Dataset ===")
    download_uci_bike_sharing_dataset(force=False)

    logger.info("=== PHASE 3: Validating Dataset ===")
    validate_dataset()

    logger.info("=== PHASE 4 & 5: Preprocessing & Feature Engineering ===")
    if force_retrain or not all(p.exists() for p in [CLEANED_HOUR_CSV, TRAIN_CSV, TEST_CSV]):
        _, train_df, _ = run_preprocessing()
    else:
        import pandas as pd
        train_df = pd.read_csv(TRAIN_CSV)
        logger.info("Processed datasets already exist; skipping preprocessing.")

    if force_retrain or not SCALER_PATH.exists():
        fit_and_save_feature_pipeline(train_df)

    logger.info("=== PHASE 6-9: Training Models ===")
    if force_retrain or not GAM_MODEL_PATH.exists():
        train_gam_model(train_df)
    else:
        logger.info("GAM model already exists at %s; skipping.", GAM_MODEL_PATH)

    if force_retrain or not LR_MODEL_PATH.exists():
        train_linear_regression_model(train_df)
    else:
        logger.info("Linear Regression model already exists at %s; skipping.", LR_MODEL_PATH)

    if force_retrain or not RF_MODEL_PATH.exists():
        train_random_forest_model(train_df)
    else:
        logger.info("Random Forest model already exists at %s; skipping.", RF_MODEL_PATH)

    if force_retrain or not GB_MODEL_PATH.exists():
        train_gradient_boosting_model(train_df)
    else:
        logger.info("Gradient Boosting model already exists at %s; skipping.", GB_MODEL_PATH)

    logger.info("=== PHASE 10: Evaluating Models ===")
    if force_retrain or not MODEL_COMPARISON_CSV.exists():
        evaluate_all_models()
    else:
        logger.info("Evaluation metrics already exist at %s; skipping.", MODEL_COMPARISON_CSV)

    logger.info("=== PHASE 11: Generating Visualizations ===")
    generate_eda_plots()
    generate_gam_partial_effect_plots()
    generate_evaluation_plots()

    logger.info("=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bike Rental Demand Prediction ML Pipeline")
    parser.add_argument("--force", action="store_true", help="Force retraining and regeneration of all artifacts")
    args = parser.parse_args()
    run_full_pipeline(force_retrain=args.force)
