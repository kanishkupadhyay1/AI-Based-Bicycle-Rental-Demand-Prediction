"""Tests for dataset loading, validation, preprocessing, and feature engineering."""

import pandas as pd

from src.config import ALL_FEATURES, CLEANED_HOUR_CSV, DATE_COLUMN, RAW_DAY_CSV, RAW_HOUR_CSV, TEST_CSV, TRAIN_CSV
from src.data.preprocess import chronological_split, clean_hourly_data
from src.data.validate_dataset import validate_dataset
from src.features.feature_engineering import get_raw_features_and_target


def test_raw_dataset_exists_and_validates():
    report = validate_dataset(RAW_HOUR_CSV, RAW_DAY_CSV)
    assert report["hour_shape"] == (17379, 17)
    assert report["day_shape"] == (731, 16)
    assert report["missing_values_hour"] == 0
    assert report["duplicate_rows_hour"] == 0


def test_clean_hourly_data_removes_leakage_and_identifiers():
    raw_df = pd.read_csv(RAW_HOUR_CSV).head(100)
    cleaned = clean_hourly_data(raw_df)
    assert "instant" not in cleaned.columns
    assert "casual" not in cleaned.columns
    assert "registered" not in cleaned.columns
    assert "cnt" in cleaned.columns
    assert list(cleaned.columns) == [DATE_COLUMN] + ALL_FEATURES + ["cnt"]


def test_chronological_split_preserves_temporal_order():
    cleaned_df = pd.read_csv(CLEANED_HOUR_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    assert len(train_df) + len(test_df) == len(cleaned_df)
    assert train_df[DATE_COLUMN].max() <= test_df[DATE_COLUMN].min()


def test_chronological_split_helper():
    df = pd.DataFrame({"val": list(range(100))})
    train, test = chronological_split(df, train_ratio=0.8)
    assert len(train) == 80
    assert len(test) == 20
    assert train["val"].max() < test["val"].min()


def test_feature_matrix_shape_and_columns():
    train_df = pd.read_csv(TRAIN_CSV).head(50)
    X, y = get_raw_features_and_target(train_df)
    assert list(X.columns) == ALL_FEATURES
    assert len(y) == 50
