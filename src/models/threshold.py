"""
Decision Threshold Optimization & Business Cost Matrix Analysis for ChurnGuard AI.
Finds optimal decision threshold minimizing total business loss (FP cost vs FN cost).
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("ThresholdOptimization")


def optimize_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    output_dir: str = "reports/figures"
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Evaluates decision thresholds from 0.10 to 0.90 to minimize business financial cost.

    Args:
        y_true: Ground truth binary target.
        y_prob: Predicted churn probabilities.
        cost_fn: Financial cost of False Negative (uncaptured churner).
        cost_fp: Financial cost of False Positive (unneeded promo incentive).
        output_dir: Output figure directory.

    Returns:
        Tuple[float, Dict[str, Any], pd.DataFrame]: Optimal threshold, best metrics summary, and full threshold results DataFrame.
    """
    thresholds = np.arange(0.10, 0.92, 0.02)
    records = []

    for tau in thresholds:
        preds = (y_prob >= tau).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        total_cost = (fn * cost_fn) + (fp * cost_fp)

        records.append({
            "threshold": round(float(tau), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "total_business_cost": round(float(total_cost), 2)
        })

    results_df = pd.DataFrame(records)

    # Find threshold minimizing financial cost
    best_row = results_df.loc[results_df["total_business_cost"].idxmin()]
    optimal_threshold = float(best_row["threshold"])

    best_summary = {
        "optimal_threshold": optimal_threshold,
        "min_business_cost": float(best_row["total_business_cost"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "cost_false_negative": cost_fn,
        "cost_false_positive": cost_fp
    }

    # Plot threshold performance and financial cost curve
    root = get_project_root()
    save_dir = root / output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(results_df["threshold"], results_df["precision"], label="Precision", color="#2980b9", lw=2)
    ax1.plot(results_df["threshold"], results_df["recall"], label="Recall", color="#e74c3c", lw=2)
    ax1.plot(results_df["threshold"], results_df["f1"], label="F1-Score", color="#2ecc71", lw=2, linestyle="--")
    ax1.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax1.set_ylabel("Metric Value (0 to 1)", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(results_df["threshold"], results_df["total_business_cost"] / 1000.0,
             label="Business Loss ($k)", color="#8e44ad", lw=2, linestyle=":")
    ax2.set_ylabel("Total Business Loss ($ Thousands)", fontsize=11, color="#8e44ad")

    ax1.axvline(x=optimal_threshold, color="black", linestyle="--",
                label=f"Optimal τ = {optimal_threshold:.2f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title(f"Threshold Optimization & Financial Loss Curve (Best τ = {optimal_threshold})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "threshold_optimization.png", dpi=300)
    plt.close()

    logger.info(f"Threshold Optimization Complete: Optimal Threshold = {optimal_threshold} | Min Business Loss = ${best_summary['min_business_cost']:,.2f}")
    return optimal_threshold, best_summary, results_df


if __name__ == "__main__":
    from .train import train_candidate_models
    res = train_candidate_models()
    # Evaluate threshold on CatBoost or Logistic Regression predictions
    model_res = res.get("CatBoost") or res.get("Logistic Regression")
    val_probs = model_res["val_probs"]

    cfg = load_config()
    root = get_project_root()
    val_df = pd.read_csv(root / cfg["data"]["val_path"])
    y_val = val_df[cfg["data"]["target_column"]].values

    opt_tau, summary, df_res = optimize_classification_threshold(y_val, val_probs)
    print("Optimal Threshold Summary:", summary)
