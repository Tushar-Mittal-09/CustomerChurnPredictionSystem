"""
SHAP Model Explainability Module for ChurnGuard AI.
Generates global feature importance plots, beeswarm plots, and individual customer risk explanations.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from .baseline import build_preprocessor
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("ModelExplainability")


def generate_shap_explanations(
    model_pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "reports/figures"
) -> Dict[str, Path]:
    """Generates global and local SHAP explanation plots.

    Args:
        model_pipeline: Fitted Scikit-learn Pipeline with preprocessor & classifier.
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        output_dir: Output figure directory.

    Returns:
        Dict[str, Path]: Map of generated SHAP plot file paths.
    """
    root = get_project_root()
    save_dir = root / output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = model_pipeline.named_steps["prep"]
    classifier = model_pipeline.named_steps["clf"]

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract feature names after OneHotEncoding
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean feature names (remove num__ and cat__)
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_test_trans.shape[1])]

    logger.info("Computing SHAP values...")
    explainer = shap.Explainer(classifier, X_train_trans)
    shap_values = explainer(X_test_trans)

    generated_files = {}

    # 1. Global Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Global SHAP Feature Importance (Bar Plot)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_bar_path = save_dir / "shap_summary_bar.png"
    plt.savefig(summary_bar_path, dpi=300)
    plt.close()
    generated_files["shap_summary_bar"] = summary_bar_path

    # 2. Global Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, show=False)
    plt.title("Global SHAP Feature Attribution (Beeswarm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = save_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    generated_files["shap_beeswarm"] = beeswarm_path

    logger.info(f"Successfully generated SHAP plots: {list(generated_files.keys())}")
    return generated_files


if __name__ == "__main__":
    from .train import train_candidate_models
    cfg = load_config()
    root = get_project_root()
    train_df = pd.read_csv(root / cfg["data"]["train_path"])
    test_df = pd.read_csv(root / cfg["data"]["test_path"])
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    res = train_candidate_models()
    best_pipe = res["Logistic Regression"]["pipeline"]  # Linear/Tree model pipeline
    generate_shap_explanations(best_pipe, X_train, X_test)
