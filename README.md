# ChurnGuard AI — Production Customer Churn Prediction & Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ChurnGuard AI** is an enterprise-ready, end-to-end Machine Learning and Retention Intelligence platform designed to identify subscribers at risk of churn, quantify decision confidence, uncover root risk drivers using SHAP explainability, and recommend targeted business retention strategies.

---

## 🏗️ System Architecture

```text
                  CUSTOMER DATA (IBM Telco Dataset)
                                │
                                ▼
                       Data Ingestion Pipeline
                                │
                                ▼
                       Automated Data Validation
                                │
                                ▼
                        Data Cleaning Engine
                                │
                                ▼
                     Domain Feature Engineering
                                │
                                ▼
                  Stratified Split (70/15/15)
                                │
                                ▼
               Candidate ML Models & Optuna Tuning
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
  Logistic Regression     Random Forest       XGBoost / CatBoost
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ▼
                     Model Comparative Evaluation
                                │
                                ▼
                  Business Threshold Optimization
                                │
                                ▼
                     SHAP Explainability Engine
                                │
                                ▼
                    Serialized Pipeline Joblib
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
         FastAPI Microservice         Streamlit Dashboard
            (Port 8000)                   (Port 8501)
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                    Docker Compose Container
```

---

## 📊 Empirical Candidate Model Performance

All models were evaluated on the held-out validation set (1,056 records) derived from the official IBM Telco Customer Churn dataset:

| Model Candidate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 0.7348 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 |
| **Logistic Regression (Best Overall)** | **0.8040** | **0.6420** | **0.5893** | **0.6145** | **0.8451** | **0.6279** |
| **Random Forest Classifier** | 0.7831 | 0.6094 | 0.5071 | 0.5536 | 0.8257 | 0.5930 |
| **XGBoost Classifier** | 0.7794 | 0.5929 | 0.5357 | 0.5629 | 0.8221 | 0.6087 |
| **CatBoost Classifier** | 0.7945 | 0.6275 | 0.5536 | 0.5882 | **0.8457** | **0.6460** |

*Note: Threshold optimization at $\tau = 0.35$ increases Recall to **> 0.82**, significantly minimizing uncaptured churn financial loss under a \$500 FN vs \$50 FP business cost matrix.*

---

## 🛠️ Tech Stack & Key Tools
- **Core Language**: Python 3.11+
- **Data & Statistics**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn, XGBoost, CatBoost
- **Hyperparameter Optimization**: Optuna (Bayesian Search)
- **Model Explainability**: SHAP (SHapley Additive exPlanations)
- **API Microservice**: FastAPI, Pydantic, Uvicorn
- **Web Dashboard**: Streamlit
- **Testing & Quality**: Pytest, Pytest-Cov
- **Containerization**: Docker, Docker Compose

---

## ⚡ Quick Start & Installation

### Option 1: Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/ChurnGuard-AI.git
cd ChurnGuard-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest

# Launch FastAPI Backend
uvicorn api.main:app --reload --port 8000

# Launch Streamlit Dashboard (in new terminal)
streamlit run dashboard/app.py
```

### Option 2: Docker Compose Setup
```bash
# Build and start all services
docker compose up --build
```
- **FastAPI Swagger Docs**: `http://localhost:8000/docs`
- **Streamlit Dashboard UI**: `http://localhost:8501`

---

## 📡 API Example Usage

### Endpoint: `POST /predict`
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

### Example Response Payload
```json
{
  "prediction": 1,
  "churn_probability": 0.6742,
  "risk_level": "HIGH",
  "threshold": 0.35,
  "top_risk_factors": [
    "Month-to-month contract",
    "High monthly charges (> $70)",
    "Short tenure (< 12 months)",
    "No tech support service",
    "Electronic check payment method"
  ],
  "recommended_action": "Offer 1-year contract discount upgrade (15% off)",
  "model_version": "1.0.0"
}
```

---

## 📁 Repository Structure
```text
ChurnGuard-AI/
├── api/                     # FastAPI backend application
│   ├── main.py
│   └── schemas.py
├── configs/                 # Centralized configuration YAML
│   └── config.yaml
├── dashboard/               # Streamlit multi-page dashboard
│   └── app.py
├── data/                    # Raw & processed data directories
│   ├── raw/
│   └── processed/
├── docs/                    # Full methodology, model card, & error analysis
│   ├── business_problem.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── model_card.md
│   ├── error_analysis.md
│   └── deployment.md
├── models/                  # Serialized Joblib pipelines & metadata
│   ├── churn_model.joblib
│   ├── preprocessing.joblib
│   └── model_metadata.json
├── reports/                 # Output figures & metrics reports
│   ├── figures/
│   └── metrics/
├── src/                     # Modular Python package
│   ├── data/                # Ingestion, validation, cleaning, EDA
│   ├── features/            # Feature engineering pipeline
│   ├── models/              # Training, baseline, evaluate, tune, threshold, predict, explain
│   └── utils/               # Config, logger, metrics utilities
├── tests/                   # Pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
