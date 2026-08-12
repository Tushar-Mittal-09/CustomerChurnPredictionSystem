"""
Model Training and Comparison Pipeline for ChurnGuard AI.
Trains and compares Logistic Regression, Random Forest, XGBoost, and CatBoost models.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import catboost as cb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb
from .baseline import build_preprocessor
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger
from ..utils.metrics import compute_classification_metrics, plot_confusion_matrix

logger = get_logger("ModelTraining")


def train_candidate_models() -> Dict[str, Dict[str, Any]]:
    """Trains Logistic Regression, Random Forest, XGBoost, and CatBoost candidates.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing trained pipelines and validation metrics.
    """
    root = get_project_root()
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]
    random_state = cfg["project"]["random_state"]

    train_df = pd.read_csv(root / cfg["data"]["train_path"])
    val_df = pd.read_csv(root / cfg["data"]["val_path"])

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    preprocessor_scaled, _, _ = build_preprocessor(X_train, scale_numeric=True)
    preprocessor_unscaled, _, _ = build_preprocessor(X_train, scale_numeric=False)

    models_config = {
        "Logistic Regression": (
            Pipeline([("prep", preprocessor_scaled), ("clf", LogisticRegression(random_state=random_state, max_iter=1000))])
        ),
        "Random Forest": (
            Pipeline([("prep", preprocessor_unscaled), ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1))])
        ),
        "XGBoost": (
            Pipeline([("prep", preprocessor_unscaled), ("clf", xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss", n_jobs=-1))])
        ),
        "CatBoost": (
            Pipeline([("prep", preprocessor_unscaled), ("clf", cb.CatBoostClassifier(iterations=200, random_seed=random_state, verbose=0))])
        )
    }

    results = {}

    for name, pipeline in models_config.items():
        logger.info(f"Training model candidate: {name}...")
        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        metrics = compute_classification_metrics(y_val.values, val_preds, val_probs)
        plot_confusion_matrix(y_val.values, val_preds, f"CM_{name}")

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "val_preds": val_preds,
            "val_probs": val_probs
        }

        logger.info(f"Candidate {name} Metrics: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

    return results


if __name__ == "__main__":
    train_candidate_models()
