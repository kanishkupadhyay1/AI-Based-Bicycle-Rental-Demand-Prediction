"""Pydantic schemas for FastAPI request and response validation."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ModelKey = Literal["gam", "linear_regression", "random_forest", "gradient_boosting"]


class PredictionInput(BaseModel):
    """Input features for hourly bicycle rental demand prediction."""

    season: int = Field(..., ge=1, le=4, description="1: Spring, 2: Summer, 3: Fall, 4: Winter")
    yr: int = Field(..., ge=0, le=1, description="Year (0: 2011, 1: 2012)")
    mnth: int = Field(..., ge=1, le=12, description="Month (1 to 12)")
    hr: int = Field(..., ge=0, le=23, description="Hour of day (0 to 23)")
    holiday: int = Field(..., ge=0, le=1, description="Whether day is a holiday (0 or 1)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0: Sunday to 6: Saturday)")
    workingday: int = Field(..., ge=0, le=1, description="Working day (0 or 1)")
    weathersit: int = Field(
        ...,
        ge=1,
        le=4,
        description="1: Clear/Partly Cloudy, 2: Mist/Cloudy, 3: Light Snow/Rain, 4: Heavy Rain/Snow",
    )
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature in Celsius")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Normalized feeling temperature")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized wind speed")


class SinglePredictionRequest(PredictionInput):
    """Request schema for predicting with a selected model."""

    model: ModelKey = Field(default="gam", description="Model identifier to use for prediction")


class SinglePredictionResponse(BaseModel):
    """Response schema for single-model prediction."""

    model: str
    model_key: ModelKey
    predicted_demand: float
    predicted_count: int
    inputs: PredictionInput


class AllModelsPredictionResponse(BaseModel):
    """Response schema for predictions from all four trained models."""

    predictions: Dict[str, float]
    rounded_predictions: Dict[str, int]
    inputs: PredictionInput


class HealthResponse(BaseModel):
    """API health status response."""

    status: str
    project: str
    models_loaded: bool
    available_models: List[str]
    dataset_loaded: bool


class ModelMetricItem(BaseModel):
    """Single model evaluation metric row."""

    model: str
    mae: float
    rmse: float
    deviance_explained: float


class MetricsResponse(BaseModel):
    """Response schema for /api/metrics."""

    metrics: List[ModelMetricItem]
    best_model_by_mae: str
    best_model_by_rmse: str
    total_records: int
    train_records: int
    test_records: int
    date_range: Dict[str, str]


class GamEffectPoint(BaseModel):
    """Single point on a GAM smooth partial effect curve."""

    x: float
    effect: float
    ci_lower: float
    ci_upper: float


class GamEffectCurve(BaseModel):
    """Smooth partial effect curve for one feature."""

    feature: str
    label: str
    term_idx: int
    points: List[GamEffectPoint]


class GamEffectsResponse(BaseModel):
    """Response schema for /api/gam-effects."""

    effects: List[GamEffectCurve]
    interpretation: str


class GenericDataResponse(BaseModel):
    """Flexible typed container for structured dataset analysis responses."""

    data: Dict[str, Any]
    message: Optional[str] = None
