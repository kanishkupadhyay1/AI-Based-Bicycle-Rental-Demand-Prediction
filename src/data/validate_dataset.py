"""Validate the downloaded UCI Bike Sharing Dataset."""

from typing import Any

import pandas as pd

from src.config import RAW_DAY_CSV, RAW_HOUR_CSV, get_logger

logger = get_logger(__name__)

EXPECTED_HOUR_COLUMNS = [
    "instant",
    "dteday",
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
    "casual",
    "registered",
    "cnt",
]

EXPECTED_DAY_COLUMNS = [col for col in EXPECTED_HOUR_COLUMNS if col != "hr"]


def validate_dataset(
    hour_path=RAW_HOUR_CSV,
    day_path=RAW_DAY_CSV,
) -> dict[str, Any]:
    """Validate raw hour.csv and day.csv files and return summary statistics.

    Raises:
        FileNotFoundError: If required raw CSV files are missing.
        ValueError: If schema, target, or data integrity checks fail.
    """
    if not hour_path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {hour_path}")
    if not day_path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {day_path}")

    hour_df = pd.read_csv(hour_path)
    day_df = pd.read_csv(day_path)

    if list(hour_df.columns) != EXPECTED_HOUR_COLUMNS:
        raise ValueError(
            f"Unexpected columns in hour.csv. Expected {EXPECTED_HOUR_COLUMNS}, got {list(hour_df.columns)}"
        )
    if list(day_df.columns) != EXPECTED_DAY_COLUMNS:
        raise ValueError(
            f"Unexpected columns in day.csv. Expected {EXPECTED_DAY_COLUMNS}, got {list(day_df.columns)}"
        )

    missing_hour = int(hour_df.isnull().sum().sum())
    missing_day = int(day_df.isnull().sum().sum())
    duplicates_hour = int(hour_df.duplicated().sum())
    duplicates_day = int(day_df.duplicated().sum())

    if (hour_df["cnt"] < 0).any():
        raise ValueError("Negative values found in target column 'cnt'.")

    if not (hour_df["casual"] + hour_df["registered"] == hour_df["cnt"]).all():
        raise ValueError("Target 'cnt' does not equal 'casual + registered' in all rows.")

    date_min = str(hour_df["dteday"].min())
    date_max = str(hour_df["dteday"].max())

    cnt_series = hour_df["cnt"]
    report: dict[str, Any] = {
        "hour_shape": hour_df.shape,
        "day_shape": day_df.shape,
        "missing_values_hour": missing_hour,
        "missing_values_day": missing_day,
        "duplicate_rows_hour": duplicates_hour,
        "duplicate_rows_day": duplicates_day,
        "date_range": {"start": date_min, "end": date_max},
        "dtypes": {col: str(dtype) for col, dtype in hour_df.dtypes.items()},
        "target_stats": {
            "count": int(cnt_series.count()),
            "mean": round(float(cnt_series.mean()), 4),
            "std": round(float(cnt_series.std()), 4),
            "min": int(cnt_series.min()),
            "q25": float(cnt_series.quantile(0.25)),
            "median": float(cnt_series.median()),
            "q75": float(cnt_series.quantile(0.75)),
            "max": int(cnt_series.max()),
            "skewness": round(float(cnt_series.skew()), 4),
        },
    }

    logger.info("Dataset Validation Summary (hour.csv):")
    logger.info("  Shape: %s (day.csv: %s)", report["hour_shape"], report["day_shape"])
    logger.info("  Date Range: %s to %s", date_min, date_max)
    logger.info("  Missing Values: hour=%d, day=%d", missing_hour, missing_day)
    logger.info("  Duplicate Rows: hour=%d, day=%d", duplicates_hour, duplicates_day)
    logger.info(
        "  Target 'cnt' Stats: mean=%.2f, std=%.2f, min=%d, median=%.1f, max=%d, skew=%.2f",
        report["target_stats"]["mean"],
        report["target_stats"]["std"],
        report["target_stats"]["min"],
        report["target_stats"]["median"],
        report["target_stats"]["max"],
        report["target_stats"]["skewness"],
    )
    return report


if __name__ == "__main__":
    validate_dataset()
