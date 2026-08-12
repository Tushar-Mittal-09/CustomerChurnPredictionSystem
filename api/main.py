"""
FastAPI REST Service for ChurnGuard AI.
Exposes real-time churn prediction, health checks, and model info endpoints.
"""
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from .schemas import (
    CustomerPredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from src.models.predict import load_serialized_pipeline, predict_churn_risk
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("API")
cfg = load_config()

app = FastAPI(
    title="ChurnGuard AI API",
    description="Production Customer Churn Prediction & Retention Intelligence Microservice",
    version=cfg["project"]["version"]
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Returns microservice health status."""
    return HealthResponse(
        status="healthy",
        service="ChurnGuard AI Prediction Service",
        version=cfg["project"]["version"]
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info() -> ModelInfoResponse:
    """Returns metadata details of serialized production model."""
    try:
        _, metadata = load_serialized_pipeline()
        return ModelInfoResponse(**metadata)
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model metadata."
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(payload: CustomerPredictionRequest) -> PredictionResponse:
    """Predicts customer churn risk level, probability, top factors, and recommended action.

    Args:
        payload: Customer features matching Telco schema.

    Returns:
        PredictionResponse: Standardized prediction response payload.
    """
    try:
        input_dict = payload.model_dump()
        df_input = pd.DataFrame([input_dict])

        results = predict_churn_risk(df_input)
        res = results[0]

        logger.info(f"Prediction generated: Prob={res['churn_probability']} | Risk={res['risk_level']}")
        return PredictionResponse(**res)
    except Exception as e:
        logger.error(f"Prediction failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error generating churn prediction."
        )
