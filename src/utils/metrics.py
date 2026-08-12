"""
Standardized Evaluation Metrics and Visualization Utility for ChurnGuard AI.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and generates evaluation plots.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    roc_curve
)
from .config import get_project_root
from .logger import get_logger

logger = get_logger("Metrics")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculates standard binary classification evaluation metrics.

    Args:
        y_true: Ground truth binary target labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).
        y_prob: Predicted probability scores for positive class (0.0 to 1.0).

    Returns:
        Dict[str, float]: Dictionary of metric names and values.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = float(auc(recalls, precisions))
        except Exception as e:
            logger.warning(f"Could not calculate ROC/PR-AUC: {e}")
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures"
) -> Path:
    """Plots and saves confusion matrix heatmap for a model.

    Args:
        y_true: True binary target.
        y_pred: Predicted binary labels.
        model_name: Name identifier of model.
        output_dir: Figure directory.

    Returns:
        Path: Saved plot filepath.
    """
    root = get_project_root()
    save_dir = root / output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    filename = save_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures"
) -> Tuple[Path, Path]:
    """Plots ROC Curve and Precision-Recall Curve.

    Args:
        y_true: True binary target.
        y_prob: Predicted probability of positive class.
        model_name: Model identifier name.
        output_dir: Output figure directory.

    Returns:
        Tuple[Path, Path]: Saved paths for ROC and PR curves.
    """
    root = get_project_root()
    save_dir = root / output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC curve (AUC = {roc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    roc_path = save_dir / f"roc_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 2. PR Curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_score = auc(recalls, precisions)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="#8e44ad", lw=2, label=f"PR curve (AUC = {pr_score:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()

    pr_path = save_dir / f"pr_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()

    return roc_path, pr_path
