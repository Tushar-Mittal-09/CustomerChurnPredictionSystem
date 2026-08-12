# ChurnGuard AI — Customer Churn Prediction & Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview
**ChurnGuard AI** is a production-quality, enterprise-grade Customer Churn Prediction and Retention Intelligence platform designed to identify at-risk customers, uncover root causes of churn using SHAP explainability, and provide tailored retention strategies.

---

## Key Features
- **End-to-End Lifecycle**: From raw data ingestion and automated validation to SHAP explainability and Docker deployment.
- **Advanced Machine Learning**: Logistic Regression, Random Forest, XGBoost, and CatBoost compared with Optuna tuning and business threshold optimization.
- **RESTful API**: FastAPI microservice with input validation (Pydantic schemas) and risk level scoring.
- **Interactive Dashboard**: Multi-page Streamlit web app for single/batch prediction, EDA performance, and SHAP explainability.
- **Production MLOps**: Centralized YAML configuration, Pytest coverage, containerization via Docker Compose.

---

## Dataset & Attribution
- **Dataset**: IBM Telco Customer Churn
- **Source URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Quick Start & Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional for containerized deployment)

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/ChurnGuard-AI.git
cd ChurnGuard-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure
Refer to [docs/methodology.md](docs/methodology.md) for full project architecture and workflow details.

---

## License
MIT License
