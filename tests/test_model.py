"""
Unit tests for metrics calculation, preprocessing pipeline, baseline models, and candidate training.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models.baseline import build_preprocessor, run_baseline_experiments
from src.models.train import train_candidate_models
from src.models.evaluate import evaluate_and_compare_models
from src.utils.metrics import compute_classification_metrics
from src.utils.config import get_project_root


def test_compute_classification_metrics():
    """Verify classification metrics calculation accuracy."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_build_preprocessor():
    """Verify ColumnTransformer handles numeric and categorical features properly."""
    root = get_project_root()
    train_df = pd.read_csv(root / "data/processed/train.csv")
    feature_cols = [c for c in train_df.columns if c not in ["Churn", "customerID"]]
    X = train_df[feature_cols]

    preprocessor, num_cols, cat_cols = build_preprocessor(X, scale_numeric=True)

    assert len(num_cols) > 0
    assert len(cat_cols) > 0
    assert len(num_cols) + len(cat_cols) == len(feature_cols)

    # Test fit_transform
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape[0] == len(X)
    assert X_trans.shape[1] >= len(feature_cols)


def test_run_baseline_experiments():
    """Verify baseline models execute and generate valid metrics dict."""
    results = run_baseline_experiments()

    assert "Majority_Class_Baseline" in results
    assert "Logistic_Regression_Baseline" in results

    lr_res = results["Logistic_Regression_Baseline"]
    assert lr_res["accuracy"] > 0.70
    assert lr_res["roc_auc"] > 0.80

    root = get_project_root()
    assert (root / "reports/metrics/baseline_metrics.json").exists()


def test_candidate_models_training_and_evaluation():
    """Verify candidate models (Logistic Regression, RF, XGBoost, CatBoost) train and evaluate."""
    summary = evaluate_and_compare_models()

    assert "Logistic Regression" in summary
    assert "Random Forest" in summary
    assert "XGBoost" in summary
    assert "CatBoost" in summary

    for model_name, metrics in summary.items():
        assert metrics["roc_auc"] > 0.75
        assert metrics["accuracy"] > 0.70

    root = get_project_root()
    assert (root / "reports/figures/model_comparison_roc.png").exists()
    assert (root / "reports/figures/model_comparison_pr.png").exists()
    assert (root / "reports/metrics/model_comparison.json").exists()
