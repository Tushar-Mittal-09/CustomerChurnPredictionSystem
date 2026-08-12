"""
Model Serialization & Inference Pipeline for ChurnGuard AI.
Exports trained model binaries (churn_model.joblib, preprocessing.joblib, model_metadata.json) and provides single/batch inference.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import catboost as cb
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .baseline import build_preprocessor
from ..features.engineering import add_engineered_features
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("ModelPredict")


def train_and_serialize_final_model(
    threshold: float = 0.35,
    model_type: str = "LogisticRegression"
) -> Tuple[Path, Path, Path]:
    """Trains final model pipeline on combined Train+Val dataset and serializes artifacts.

    Args:
        threshold: Optimal decision threshold.
        model_type: Type of classifier to train.

    Returns:
        Tuple[Path, Path, Path]: Paths to model joblib, preprocessor joblib, and metadata json.
    """
    root = get_project_root()
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]
    random_state = cfg["project"]["random_state"]

    train_df = pd.read_csv(root / cfg["data"]["train_path"])
    val_df = pd.read_csv(root / cfg["data"]["val_path"])
    full_train = pd.concat([train_df, val_df], ignore_index=True)

    feature_cols = [c for c in full_train.columns if c not in [target_col, id_col]]
    X_full, y_full = full_train[feature_cols], full_train[target_col]

    preprocessor, num_cols, cat_cols = build_preprocessor(X_full, scale_numeric=True)

    if model_type == "CatBoost":
        classifier = cb.CatBoostClassifier(iterations=200, random_seed=random_state, verbose=0)
    else:
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)

    model_pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", classifier)
    ])

    logger.info(f"Fitting final {model_type} pipeline on {len(full_train)} records...")
    model_pipeline.fit(X_full, y_full)

    models_dir = root / cfg["model"]["path"]
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / cfg["model"]["model_filename"]
    prep_path = models_dir / cfg["model"]["preprocessing_filename"]
    meta_path = models_dir / cfg["model"]["metadata_filename"]

    joblib.dump(model_pipeline, model_path)
    joblib.dump(preprocessor, prep_path)

    metadata = {
        "model_name": f"ChurnGuard AI — {model_type}",
        "model_version": cfg["project"]["version"],
        "threshold": threshold,
        "features": feature_cols,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "training_date": datetime.now().isoformat(),
        "n_training_samples": len(full_train)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully serialized final model pipeline to {model_path}")
    return model_path, prep_path, meta_path


def load_serialized_pipeline() -> Tuple[Any, Dict[str, Any]]:
    """Loads serialized pipeline and metadata.

    Returns:
        Tuple[Any, Dict[str, Any]]: Loaded Pipeline object and metadata dictionary.
    """
    root = get_project_root()
    cfg = load_config()
    models_dir = root / cfg["model"]["path"]

    model_path = models_dir / cfg["model"]["model_filename"]
    meta_path = models_dir / cfg["model"]["metadata_filename"]

    if not model_path.exists():
        logger.info("Serialized model not found. Retraining final model...")
        train_and_serialize_final_model()

    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata


def predict_churn_risk(
    input_data: pd.DataFrame,
    threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Generates churn prediction, risk level, and risk factors for input customer records.

    Args:
        input_data: Raw or feature-engineered input DataFrame.
        threshold: Optional custom decision threshold.

    Returns:
        List[Dict[str, Any]]: Prediction results per customer.
    """
    pipeline, metadata = load_serialized_pipeline()
    effective_threshold = threshold if threshold is not None else metadata.get("threshold", 0.35)

    # Apply feature engineering if derived features missing
    if "total_services" not in input_data.columns:
        df_proc = add_engineered_features(input_data)
    else:
        df_proc = input_data.copy()

    probs = pipeline.predict_proba(df_proc)[:, 1]

    results = []
    for idx, prob in enumerate(probs):
        prob_val = float(prob)
        prediction = 1 if prob_val >= effective_threshold else 0

        # Risk Level Determination
        if prob_val < 0.30:
            risk_level = "LOW"
        elif prob_val < 0.60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Risk Factor Analysis
        row = df_proc.iloc[idx]
        top_factors = []
        if str(row.get("Contract", "")) == "Month-to-month":
            top_factors.append("Month-to-month contract")
        if float(row.get("MonthlyCharges", 0)) > 70.0:
            top_factors.append("High monthly charges (> $70)")
        if float(row.get("tenure", 0)) < 12:
            top_factors.append("Short tenure (< 12 months)")
        if str(row.get("TechSupport", "")) == "No":
            top_factors.append("No tech support service")
        if str(row.get("PaymentMethod", "")) == "Electronic check":
            top_factors.append("Electronic check payment method")

        if not top_factors:
            top_factors.append("Standard behavioral pattern")

        # Business Retention Recommendation Rule
        if risk_level == "HIGH":
            if "Month-to-month contract" in top_factors:
                recommendation = "Offer 1-year contract discount upgrade (15% off)"
            elif "High monthly charges (> $70)" in top_factors:
                recommendation = "Review tier pricing and offer custom bundle discount"
            else:
                recommendation = "Prioritize for proactive customer success outreach"
        elif risk_level == "MEDIUM":
            recommendation = "Send automated engagement email with tech support highlight"
        else:
            recommendation = "Maintain standard customer lifecycle engagement"

        results.append({
            "churn_probability": round(prob_val, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "threshold": effective_threshold,
            "top_risk_factors": top_factors,
            "recommended_action": recommendation,
            "model_version": metadata.get("model_version", "1.0.0")
        })

    return results


if __name__ == "__main__":
    train_and_serialize_final_model()
