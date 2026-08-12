"""
Model Evaluation and Comparison Pipeline for ChurnGuard AI.
Compares models on ROC/PR curves and exports performance metric reports.
"""
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from .train import train_candidate_models
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("ModelEvaluate")


def evaluate_and_compare_models() -> Dict[str, Dict[str, float]]:
    """Evaluates all candidate models, generates comparative ROC/PR plots, and saves results.

    Returns:
        Dict[str, Dict[str, float]]: Summary metrics for each model.
    """
    root = get_project_root()
    cfg = load_config()
    target_col = cfg["data"]["target_column"]

    val_df = pd.read_csv(root / cfg["data"]["val_path"])
    y_val = val_df[target_col].values

    results_dict = train_candidate_models()

    summary_metrics = {}
    fig_dir = root / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance")

    # 2. Combined PR Curve
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    baseline_pr = float(y_val.mean())
    ax_pr.axhline(y=baseline_pr, color="grey", linestyle="--", label=f"Baseline ({baseline_pr:.2f})")

    colors = {
        "Logistic Regression": "#2980b9",
        "Random Forest": "#2ecc71",
        "XGBoost": "#e67e22",
        "CatBoost": "#8e44ad"
    }

    for name, res in results_dict.items():
        metrics = res["metrics"]
        summary_metrics[name] = metrics
        val_probs = res["val_probs"]

        # ROC
        fpr, tpr, _ = roc_curve(y_val, val_probs)
        ax_roc.plot(fpr, tpr, color=colors.get(name, "blue"), lw=2,
                     label=f"{name} (AUC = {metrics['roc_auc']:.3f})")

        # PR
        precisions, recalls, _ = precision_recall_curve(y_val, val_probs)
        ax_pr.plot(recalls, precisions, color=colors.get(name, "blue"), lw=2,
                    label=f"{name} (AUC = {metrics['pr_auc']:.3f})")

    # Finalize ROC plot
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("Model Comparison — ROC Curves", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(fig_dir / "model_comparison_roc.png", dpi=300)
    plt.close(fig_roc)

    # Finalize PR plot
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Model Comparison — Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(fig_dir / "model_comparison_pr.png", dpi=300)
    plt.close(fig_pr)

    # Save summary metrics JSON
    metrics_dir = root / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"Model comparison complete. Results saved to {metrics_dir / 'model_comparison.json'}")
    return summary_metrics


if __name__ == "__main__":
    evaluate_and_compare_models()
