"""
Optuna Bayesian Hyperparameter Tuning & Class Imbalance Handling for ChurnGuard AI.
Tunes XGBoost, CatBoost, and Random Forest models using cross-validation on the training set.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import catboost as cb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
from .baseline import build_preprocessor
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger("HyperparameterTuning")


def tune_xgboost(X_trans: np.ndarray, y: pd.Series, n_trials: int = 15) -> Tuple[Dict[str, Any], float]:
    """Tunes XGBoost hyperparameters using Optuna on preprocessed training data.

    Args:
        X_trans: Preprocessed feature array.
        y: Target series.
        n_trials: Number of Bayesian search trials.

    Returns:
        Tuple[Dict[str, Any], float]: Best parameters and best ROC-AUC score.
    """
    scale_pos_weight = float((len(y) - y.sum()) / y.sum())

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, scale_pos_weight]),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(xgb.XGBClassifier(**params), X_trans, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"XGBoost Tuning Complete. Best ROC-AUC: {study.best_value:.4f}")
    return study.best_params, float(study.best_value)


def tune_catboost(X_trans: np.ndarray, y: pd.Series, n_trials: int = 10) -> Tuple[Dict[str, Any], float]:
    """Tunes CatBoost hyperparameters using Optuna on preprocessed training data.

    Args:
        X_trans: Preprocessed feature array.
        y: Target series.
        n_trials: Number of trials.

    Returns:
        Tuple[Dict[str, Any], float]: Best parameters and best ROC-AUC score.
    """
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 80, 200),
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["None", "Balanced"]),
            "random_seed": 42,
            "verbose": 0
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(cb.CatBoostClassifier(**params), X_trans, y, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"CatBoost Tuning Complete. Best ROC-AUC: {study.best_value:.4f}")
    return study.best_params, float(study.best_value)


def run_hyperparameter_optimization(n_trials: int = 10) -> Dict[str, Any]:
    """Executes Optuna tuning for top models and exports tuning report.

    Args:
        n_trials: Number of tuning iterations per model.

    Returns:
        Dict[str, Any]: Combined tuning results dictionary.
    """
    root = get_project_root()
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]

    train_df = pd.read_csv(root / cfg["data"]["train_path"])
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    X_train, y_train = train_df[feature_cols], train_df[target_col]

    preprocessor, _, _ = build_preprocessor(X_train, scale_numeric=False)
    X_trans = preprocessor.fit_transform(X_train)

    logger.info("Starting Bayesian Hyperparameter Tuning with Optuna...")
    xgb_params, xgb_score = tune_xgboost(X_trans, y_train, n_trials=n_trials)
    cb_params, cb_score = tune_catboost(X_trans, y_train, n_trials=n_trials)

    results = {
        "XGBoost": {"best_params": xgb_params, "best_cv_roc_auc": xgb_score},
        "CatBoost": {"best_params": cb_params, "best_cv_roc_auc": cb_score}
    }

    metrics_dir = root / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "optuna_tuning.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved Optuna hyperparameter tuning report to {metrics_dir / 'optuna_tuning.json'}")
    return results


if __name__ == "__main__":
    run_hyperparameter_optimization(n_trials=10)
