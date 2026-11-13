"""Preprocess the UCI Bike Sharing hourly dataset and create chronological splits."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import (
    ALL_FEATURES,
    CLEANED_HOUR_CSV,
    DATE_COLUMN,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    RAW_HOUR_CSV,
    TARGET_COLUMN,
    TEST_CSV,
    TRAIN_CSV,
    TRAIN_SPLIT_RATIO,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

INTEGER_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "cnt",
]

FLOAT_COLUMNS = ["temp", "atemp", "hum", "windspeed"]


def clean_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw hourly bike sharing DataFrame without data leakage.

    - Drops irrelevant identifier columns ('instant') and target-component leakage
      columns ('casual', 'registered').
    - Sorts chronologically by ('dteday', 'hr').
    - Enforces expected dtypes and drops exact duplicates if present.
    - Handles missing values only if present.
    """
    cleaned = df.copy()

    cols_to_drop = [c for c in IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS if c in cleaned.columns]
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    if cleaned.isnull().any().any():
        missing_count = int(cleaned.isnull().sum().sum())
        logger.warning("Found %d missing values; dropping incomplete rows.", missing_count)
        cleaned = cleaned.dropna().reset_index(drop=True)

    dup_count = int(cleaned.duplicated().sum())
    if dup_count > 0:
        logger.warning("Found %d duplicate rows; removing duplicates.", dup_count)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for col in INTEGER_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(int)
    for col in FLOAT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(float)

    if DATE_COLUMN in cleaned.columns and "hr" in cleaned.columns:
        cleaned = cleaned.sort_values(by=[DATE_COLUMN, "hr"]).reset_index(drop=True)

    expected_cols = [DATE_COLUMN] + ALL_FEATURES + [TARGET_COLUMN]
    cleaned = cleaned[expected_cols]
    return cleaned


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically into training and testing sets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def run_preprocessing(
    raw_path: Path = RAW_HOUR_CSV,
    cleaned_path: Path = CLEANED_HOUR_CSV,
    train_path: Path = TRAIN_CSV,
    test_path: Path = TEST_CSV,
    train_ratio: float = TRAIN_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute full preprocessing pipeline and save processed CSV files."""
    ensure_directories()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run download_dataset.py first.")

    logger.info("Loading raw hourly dataset from %s", raw_path)
    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_hourly_data(raw_df)

    train_df, test_df = chronological_split(cleaned_df, train_ratio=train_ratio)

    cleaned_df.to_csv(cleaned_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        "Preprocessing complete: total=%d, train=%d (%s to %s), test=%d (%s to %s)",
        len(cleaned_df),
        len(train_df),
        train_df[DATE_COLUMN].min(),
        train_df[DATE_COLUMN].max(),
        len(test_df),
        test_df[DATE_COLUMN].min(),
        test_df[DATE_COLUMN].max(),
    )
    return cleaned_df, train_df, test_df


if __name__ == "__main__":
    run_preprocessing()
