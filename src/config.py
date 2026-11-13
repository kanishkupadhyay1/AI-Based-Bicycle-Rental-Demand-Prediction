"""Centralized paths and project configuration."""

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"

RAW_HOUR_CSV = RAW_DATA_DIR / "hour.csv"
RAW_DAY_CSV = RAW_DATA_DIR / "day.csv"
RAW_README_TXT = RAW_DATA_DIR / "Readme.txt"

CLEANED_HOUR_CSV = PROCESSED_DATA_DIR / "cleaned_hour.csv"
TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

GAM_MODEL_PATH = MODELS_DIR / "gam_model.pkl"
LR_MODEL_PATH = MODELS_DIR / "linear_regression_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
GB_MODEL_PATH = MODELS_DIR / "gradient_boosting_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"

MODEL_COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"
TEST_PREDICTIONS_CSV = PREDICTIONS_DIR / "test_predictions.csv"

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
UCI_DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

TARGET_COLUMN = "cnt"
LEAKAGE_COLUMNS = ["casual", "registered"]
IDENTIFIER_COLUMNS = ["instant"]
DATE_COLUMN = "dteday"

CONTINUOUS_FEATURES = ["temp", "atemp", "hum", "windspeed"]
TEMPORAL_CONTINUOUS_FEATURES = ["hr", "mnth"]
CATEGORICAL_FEATURES = ["season", "yr", "holiday", "weekday", "workingday", "weathersit"]
ALL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

TRAIN_SPLIT_RATIO = 0.80


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
