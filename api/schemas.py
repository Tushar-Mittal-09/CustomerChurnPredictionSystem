"""
Pydantic Request & Response Validation Schemas for ChurnGuard AI FastAPI Service.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerPredictionRequest(BaseModel):
    gender: str = Field(default="Female", description="Customer gender ('Male', 'Female')")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior citizen flag (0 or 1)")
    Partner: str = Field(default="No", description="Partner flag ('Yes', 'No')")
    Dependents: str = Field(default="No", description="Dependents flag ('Yes', 'No')")
    tenure: int = Field(default=8, ge=0, le=120, description="Tenure in months (0 to 120)")
    PhoneService: str = Field(default="Yes", description="Phone service ('Yes', 'No')")
    MultipleLines: str = Field(default="No", description="Multiple lines ('Yes', 'No', 'No phone service')")
    InternetService: str = Field(default="Fiber optic", description="Internet provider ('DSL', 'Fiber optic', 'No')")
    OnlineSecurity: str = Field(default="No", description="Online security ('Yes', 'No', 'No internet service')")
    OnlineBackup: str = Field(default="No", description="Online backup ('Yes', 'No', 'No internet service')")
    DeviceProtection: str = Field(default="No", description="Device protection ('Yes', 'No', 'No internet service')")
    TechSupport: str = Field(default="No", description="Tech support ('Yes', 'No', 'No internet service')")
    StreamingTV: str = Field(default="No", description="Streaming TV ('Yes', 'No', 'No internet service')")
    StreamingMovies: str = Field(default="No", description="Streaming Movies ('Yes', 'No', 'No internet service')")
    Contract: str = Field(default="Month-to-month", description="Contract type ('Month-to-month', 'One year', 'Two year')")
    PaperlessBilling: str = Field(default="Yes", description="Paperless billing ('Yes', 'No')")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')")
    MonthlyCharges: float = Field(default=89.5, ge=0.0, le=500.0, description="Monthly charge in USD")
    TotalCharges: float = Field(default=716.0, ge=0.0, le=20000.0, description="Total historical charges in USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 8,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.5,
                "TotalCharges": 716.0
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary churn prediction: 1 (Churn) or 0 (Retained)")
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk tier classification: 'LOW', 'MEDIUM', 'HIGH'")
    threshold: float = Field(..., description="Decision threshold applied")
    top_risk_factors: List[str] = Field(..., description="Primary features driving churn risk")
    recommended_action: str = Field(..., description="Rule-based retention campaign recommendation")
    model_version: str = Field(..., description="Active model release version")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    features: List[str]
    training_date: str
    n_training_samples: int
