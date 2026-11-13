"""Service layer for loading trained models, dataset analysis, and serving predictions."""

import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    CLEANED_HOUR_CSV,
    DATE_COLUMN,
    FEATURE_CONFIG_PATH,
    GAM_MODEL_PATH,
    GB_MODEL_PATH,
    LR_MODEL_PATH,
    MODEL_COMPARISON_CSV,
    RF_MODEL_PATH,
    SCALER_PATH,
    TEST_CSV,
    TEST_PREDICTIONS_CSV,
    TRAIN_CSV,
    get_logger,
)
from src.visualization.gam_effect_plots import extract_gam_partial_effects

logger = get_logger(__name__)

MODEL_DISPLAY_NAMES = {
    "gam": "Generalized Additive Model (GAM)",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
}


class PredictionService:
    """Loads persisted models and real dataset artifacts to serve API requests."""

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.preprocessor: Any = None
        self.feature_config: Dict[str, Any] = {}
        self.gam_effects_cache: list[dict[str, Any]] | None = None
        self.models_loaded: bool = False

    def load_artifacts(self) -> None:
        """Load trained models, preprocessor, and feature metadata from disk."""
        required_paths = {
            "gam": GAM_MODEL_PATH,
            "linear_regression": LR_MODEL_PATH,
            "random_forest": RF_MODEL_PATH,
            "gradient_boosting": GB_MODEL_PATH,
            "preprocessor": SCALER_PATH,
            "feature_config": FEATURE_CONFIG_PATH,
        }
        missing = [str(p) for p in required_paths.values() if not p.exists()]
        if missing:
            self.models_loaded = False
            raise FileNotFoundError(
                f"Required model artifacts are missing: {missing}. Run the training pipeline first."
            )

        self.models = {
            "gam": joblib.load(GAM_MODEL_PATH),
            "linear_regression": joblib.load(LR_MODEL_PATH),
            "random_forest": joblib.load(RF_MODEL_PATH),
            "gradient_boosting": joblib.load(GB_MODEL_PATH),
        }
        self.preprocessor = joblib.load(SCALER_PATH)
        self.feature_config = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
        self.gam_effects_cache = extract_gam_partial_effects(GAM_MODEL_PATH)
        self.models_loaded = True
        logger.info("Successfully loaded all 4 trained models and feature artifacts.")

    def _ensure_loaded(self) -> None:
        if not self.models_loaded:
            self.load_artifacts()

    def predict_single(self, model_key: str, features: Dict[str, Any]) -> float:
        """Run inference for a single model and return non-negative predicted count."""
        self._ensure_loaded()
        if model_key not in self.models:
            raise ValueError(f"Unsupported model key: {model_key}")

        input_df = pd.DataFrame([{col: features[col] for col in ALL_FEATURES}])

        if model_key == "gam":
            raw_pred = float(self.models["gam"].predict(input_df[ALL_FEATURES].to_numpy(dtype=float))[0])
        else:
            X_transformed = self.preprocessor.transform(input_df)
            raw_pred = float(self.models[model_key].predict(X_transformed)[0])

        return round(max(0.0, raw_pred), 2)

    def predict_all(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Run inference across all four trained models."""
        return {
            MODEL_DISPLAY_NAMES[key]: self.predict_single(key, features)
            for key in ["gam", "linear_regression", "random_forest", "gradient_boosting"]
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Load actual evaluation metrics and dataset split counts."""
        for p in [MODEL_COMPARISON_CSV, CLEANED_HOUR_CSV, TRAIN_CSV, TEST_CSV]:
            if not p.exists():
                raise FileNotFoundError(f"Required artifact missing: {p}")

        metrics_df = pd.read_csv(MODEL_COMPARISON_CSV)
        cleaned_df = pd.read_csv(CLEANED_HOUR_CSV)
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)

        best_mae_row = metrics_df.loc[metrics_df["mae"].idxmin()]
        best_rmse_row = metrics_df.loc[metrics_df["rmse"].idxmin()]

        return {
            "metrics": metrics_df.to_dict(orient="records"),
            "best_model_by_mae": str(best_mae_row["model"]),
            "best_model_by_rmse": str(best_rmse_row["model"]),
            "total_records": int(len(cleaned_df)),
            "train_records": int(len(train_df)),
            "test_records": int(len(test_df)),
            "date_range": {
                "start": str(cleaned_df[DATE_COLUMN].min()),
                "end": str(cleaned_df[DATE_COLUMN].max()),
            },
        }

    def get_model_comparison_details(self) -> Dict[str, Any]:
        """Return metrics, sample actual-vs-predicted series, and residual summary from test_predictions.csv."""
        if not TEST_PREDICTIONS_CSV.exists():
            raise FileNotFoundError(f"Test predictions file missing: {TEST_PREDICTIONS_CSV}")

        summary = self.get_metrics_summary()
        pred_df = pd.read_csv(TEST_PREDICTIONS_CSV)

        # Provide a chronological slice (last 168 hours = 7 days of test set) for clean time-series plotting
        recent_slice = pred_df.tail(168).copy()
        recent_slice["label"] = recent_slice[DATE_COLUMN].astype(str) + " H" + recent_slice["hr"].astype(str)

        # Also provide evenly sampled scatter points across the full test set (max 300 points)
        step = max(1, len(pred_df) // 300)
        scatter_df = pred_df.iloc[::step].copy()

        # Compute residual summary statistics per model
        model_col_map = {
            "Generalized Additive Model (GAM)": "gam_pred",
            "Linear Regression": "linear_regression_pred",
            "Random Forest Regressor": "random_forest_pred",
            "Gradient Boosting Regressor": "gradient_boosting_pred",
        }
        residual_stats = []
        for model_name, col in model_col_map.items():
            res = pred_df["actual_cnt"] - pred_df[col]
            residual_stats.append(
                {
                    "model": model_name,
                    "mean_residual": round(float(res.mean()), 2),
                    "std_residual": round(float(res.std()), 2),
                    "median_abs_error": round(float(res.abs().median()), 2),
                    "p90_abs_error": round(float(res.abs().quantile(0.90)), 2),
                }
            )

        return {
            **summary,
            "recent_time_series": recent_slice[
                [
                    "label",
                    "actual_cnt",
                    "gam_pred",
                    "linear_regression_pred",
                    "random_forest_pred",
                    "gradient_boosting_pred",
                ]
            ].to_dict(orient="records"),
            "scatter_points": scatter_df[
                [
                    "actual_cnt",
                    "gam_pred",
                    "linear_regression_pred",
                    "random_forest_pred",
                    "gradient_boosting_pred",
                ]
            ].to_dict(orient="records"),
            "residual_stats": residual_stats,
        }

    def get_demand_analysis(self) -> Dict[str, Any]:
        """Compute real EDA demand summaries from cleaned_hour.csv."""
        if not CLEANED_HOUR_CSV.exists():
            raise FileNotFoundError(f"Cleaned dataset missing: {CLEANED_HOUR_CSV}")

        df = pd.read_csv(CLEANED_HOUR_CSV)

        # Hourly demand by workingday
        hourly_pivot = (
            df.groupby(["hr", "workingday"])["cnt"]
            .mean()
            .unstack(fill_value=0)
            .reset_index()
        )
        hourly_demand = [
            {
                "hr": int(row["hr"]),
                "working_day": round(float(row.get(1, 0.0)), 2),
                "non_working_day": round(float(row.get(0, 0.0)), 2),
            }
            for _, row in hourly_pivot.iterrows()
        ]

        # Weather situation demand
        weather_labels = {
            1: "Clear / Partly Cloudy",
            2: "Mist / Cloudy",
            3: "Light Snow / Rain",
            4: "Heavy Rain / Snow",
        }
        weather_grp = df.groupby("weathersit")["cnt"].agg(["mean", "count"]).reset_index()
        weather_demand = [
            {
                "weathersit": int(row["weathersit"]),
                "label": weather_labels.get(int(row["weathersit"]), f"Code {int(row['weathersit'])}"),
                "avg_demand": round(float(row["mean"]), 2),
                "observations": int(row["count"]),
            }
            for _, row in weather_grp.iterrows()
        ]

        # Seasonal demand
        season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        season_grp = df.groupby("season")["cnt"].mean().reset_index()
        seasonal_demand = [
            {
                "season": int(row["season"]),
                "label": season_labels.get(int(row["season"]), str(row["season"])),
                "avg_demand": round(float(row["cnt"]), 2),
            }
            for _, row in season_grp.iterrows()
        ]

        # Monthly demand
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_grp = df.groupby("mnth")["cnt"].mean().reset_index()
        monthly_demand = [
            {
                "mnth": int(row["mnth"]),
                "label": month_names[int(row["mnth"]) - 1],
                "avg_demand": round(float(row["cnt"]), 2),
            }
            for _, row in monthly_grp.iterrows()
        ]

        # Temperature binned demand
        temp_bins = pd.cut(df["temp"], bins=10)
        temp_grp = df.groupby(temp_bins, observed=False)["cnt"].mean().reset_index()
        temperature_demand = [
            {
                "temp": round(float(row["temp"].mid), 2),
                "avg_demand": round(float(row["cnt"]), 2) if pd.notna(row["cnt"]) else 0.0,
            }
            for _, row in temp_grp.iterrows()
        ]

        return {
            "total_records": int(len(df)),
            "date_range": {
                "start": str(df[DATE_COLUMN].min()),
                "end": str(df[DATE_COLUMN].max()),
            },
            "target_summary": {
                "mean": round(float(df["cnt"].mean()), 2),
                "median": round(float(df["cnt"].median()), 2),
                "min": int(df["cnt"].min()),
                "max": int(df["cnt"].max()),
                "std": round(float(df["cnt"].std()), 2),
            },
            "hourly_demand": hourly_demand,
            "weather_demand": weather_demand,
            "seasonal_demand": seasonal_demand,
            "monthly_demand": monthly_demand,
            "temperature_demand": temperature_demand,
        }

    def get_gam_effects(self) -> list[dict[str, Any]]:
        """Return cached or freshly extracted GAM partial dependence curves."""
        self._ensure_loaded()
        if self.gam_effects_cache is None:
            self.gam_effects_cache = extract_gam_partial_effects(GAM_MODEL_PATH)
        return self.gam_effects_cache

    def get_feature_info(self) -> Dict[str, Any]:
        """Return dataset-derived feature ranges and defaults."""
        self._ensure_loaded()
        return self.feature_config
