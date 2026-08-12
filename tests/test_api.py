"""
Unit & Integration Tests for FastAPI Endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify /health endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ChurnGuard" in data["service"]


def test_model_info_endpoint():
    """Verify /model-info returns metadata details."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "threshold" in data
    assert "features" in data


def test_predict_endpoint_success():
    """Verify /predict returns valid churn prediction for valid customer payload."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
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
        "TotalCharges": 268.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["top_risk_factors"], list)
    assert len(data["recommended_action"]) > 0


def test_predict_endpoint_invalid_payload():
    """Verify /predict returns 422 Unprocessable Entity for invalid negative tenure."""
    payload = {
        "gender": "Female",
        "tenure": -50  # Invalid negative tenure violates Pydantic ge=0 constraint
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
