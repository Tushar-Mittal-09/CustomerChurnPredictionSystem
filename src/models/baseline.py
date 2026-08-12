"""
Preprocessing Pipeline and Baseline Models for ChurnGuard AI.
Implements Sklearn ColumnTransformer preprocessing and evaluates Majority Class & Logistic Regression baselines.
"""
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger
from ..utils.metrics import compute_classification_metrics, plot_confusion_matrix, plot_roc_pr_curves

logger = get_logger("BaselineModels")


def build_preprocessor(
    df: pd.DataFrame,
    scale_numeric: bool = True
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Builds a scikit-learn ColumnTransformer preprocessing pipeline.

    Args:
        df: Training feature DataFrame (excluding target and ID columns).
        scale_numeric: If True, applies StandardScaler to numerical features.

    Returns:
        Tuple[ColumnTransformer, List[str], List[str]]: Preprocessor, num_cols, cat_cols.
    """
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]

    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]

    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    # Numerical Transformer
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    # Categorical Transformer
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    logger.info(f"Built ColumnTransformer with {len(num_cols)} numerical and {len(cat_cols)} categorical features.")
    return preprocessor, num_cols, cat_cols


def run_baseline_experiments() -> Dict[str, Any]:
    """Trains Baseline 1 (Majority Class) and Baseline 2 (Logistic Regression) and records metrics.

    Returns:
        Dict[str, Any]: Metrics dictionary for baseline models.
    """
    root = get_project_root()
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]
    random_state = cfg["project"]["random_state"]

    train_path = root / cfg["data"]["train_path"]
    val_path = root / cfg["data"]["val_path"]

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    preprocessor, _, _ = build_preprocessor(X_train, scale_numeric=True)

    results = {}

    # 1. Baseline 1: Majority Class Predictor
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_preds = dummy_model.predict(X_val)
    dummy_probs = dummy_model.predict_proba(X_val)[:, 1]
    dummy_metrics = compute_classification_metrics(y_val.values, dummy_preds, dummy_probs)
    results["Majority_Class_Baseline"] = dummy_metrics
    logger.info(f"Baseline 1 (Majority Class): Accuracy={dummy_metrics['accuracy']:.4f}, F1={dummy_metrics['f1']:.4f}, ROC-AUC={dummy_metrics['roc_auc']:.4f}")

    # 2. Baseline 2: Logistic Regression Pipeline
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=random_state, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_val)
    lr_probs = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_metrics = compute_classification_metrics(y_val.values, lr_preds, lr_probs)
    results["Logistic_Regression_Baseline"] = lr_metrics
    logger.info(f"Baseline 2 (Logistic Regression): Accuracy={lr_metrics['accuracy']:.4f}, F1={lr_metrics['f1']:.4f}, ROC-AUC={lr_metrics['roc_auc']:.4f}")

    # Save plots for Logistic Regression Baseline
    plot_confusion_matrix(y_val.values, lr_preds, "Logistic Regression Baseline")
    plot_roc_pr_curves(y_val.values, lr_probs, "Logistic Regression Baseline")

    # Save metrics JSON
    metrics_dir = root / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_baseline_experiments()
