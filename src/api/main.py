"""FastAPI backend application for AI-Based Bicycle Rental Demand Prediction."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.prediction_service import MODEL_DISPLAY_NAMES, PredictionService
from src.api.schemas import (
    AllModelsPredictionResponse,
    GamEffectsResponse,
    HealthResponse,
    MetricsResponse,
    PredictionInput,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from src.config import CLEANED_HOUR_CSV, get_logger

logger = get_logger(__name__)
service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained ML models and artifacts on startup."""
    try:
        service.load_artifacts()
    except FileNotFoundError as exc:
        logger.warning("Startup model load deferred/failed: %s", exc)
    yield


app = FastAPI(
    title="AI-Based Bicycle Rental Demand Prediction API",
    description=(
        "FastAPI backend serving Generalized Additive Model (PoissonGAM), Linear Regression, "
        "Random Forest, and Gradient Boosting predictions and real UCI Bike Sharing Dataset analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Return API health and model availability status."""
    return HealthResponse(
        status="healthy",
        project="AI-Based Bicycle Rental Demand Prediction (GAM)",
        models_loaded=service.models_loaded,
        available_models=list(MODEL_DISPLAY_NAMES.values()) if service.models_loaded else [],
        dataset_loaded=CLEANED_HOUR_CSV.exists(),
    )


@app.get("/api/metrics", response_model=MetricsResponse, tags=["Evaluation"])
def get_metrics() -> MetricsResponse:
    """Return actual model evaluation metrics from model_comparison.csv."""
    try:
        summary = service.get_metrics_summary()
        return MetricsResponse(**summary)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error retrieving metrics")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/model-comparison", tags=["Evaluation"])
def get_model_comparison():
    """Return model comparison metrics, test predictions slice, and residual statistics."""
    try:
        return service.get_model_comparison_details()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error retrieving model comparison details")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/demand-analysis", tags=["EDA"])
def get_demand_analysis():
    """Return real demand analysis aggregated from cleaned_hour.csv."""
    try:
        return service.get_demand_analysis()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error computing demand analysis")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/gam-effects", response_model=GamEffectsResponse, tags=["GAM Analysis"])
def get_gam_effects() -> GamEffectsResponse:
    """Return zero-centered PoissonGAM smooth partial dependence curves and 95% CIs."""
    try:
        effects = service.get_gam_effects()
        return GamEffectsResponse(
            effects=effects,
            interpretation=(
                "Each curve shows the zero-centered smooth partial effect of a feature on the "
                "log expected hourly bicycle rental count (holding all other features fixed). "
                "Positive values indicate above-average demand contribution; negative values indicate "
                "below-average demand contribution. Shaded bands represent 95% confidence intervals."
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error extracting GAM partial effects")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/feature-info", tags=["Metadata"])
def get_feature_info():
    """Return feature bounds, defaults, and metadata derived from the real dataset."""
    try:
        return service.get_feature_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error retrieving feature metadata")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
def predict_single(request: SinglePredictionRequest) -> SinglePredictionResponse:
    """Predict hourly bicycle rental demand using the selected trained model."""
    try:
        input_dict = request.model_dump(exclude={"model"})
        pred_value = service.predict_single(request.model, input_dict)
        return SinglePredictionResponse(
            model=MODEL_DISPLAY_NAMES[request.model],
            model_key=request.model,
            predicted_demand=pred_value,
            predicted_count=int(round(pred_value)),
            inputs=PredictionInput(**input_dict),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict-all", response_model=AllModelsPredictionResponse, tags=["Prediction"])
def predict_all_models(request: PredictionInput) -> AllModelsPredictionResponse:
    """Predict hourly bicycle rental demand across all four trained models."""
    try:
        input_dict = request.model_dump()
        preds = service.predict_all(input_dict)
        rounded = {k: int(round(v)) for k, v in preds.items()}
        return AllModelsPredictionResponse(
            predictions=preds,
            rounded_predictions=rounded,
            inputs=request,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Multi-model prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
