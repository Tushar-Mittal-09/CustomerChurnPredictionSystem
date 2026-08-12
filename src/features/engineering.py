"""
Feature Engineering Pipeline for ChurnGuard AI.
Derives business features, conducts target leakage checks, and performs stratified Train/Val/Test splitting.
"""
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..data.cleaning import clean_dataset
from ..data.ingestion import ingest_raw_data
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("FeatureEngineering")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-derived features for Telco customer churn prediction.

    Args:
        df: Cleaned input DataFrame.

    Returns:
        pd.DataFrame: DataFrame augmented with engineered features.
    """
    df_feat = df.copy()

    # 1. Total active add-on services count
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Count how many of these services equal 'Yes'
    df_feat["total_services"] = 0
    for col in service_cols:
        if col in df_feat.columns:
            df_feat["total_services"] += (df_feat[col] == "Yes").astype(int)

    # Add Phone & Internet count
    if "PhoneService" in df_feat.columns:
        df_feat["total_services"] += (df_feat["PhoneService"] == "Yes").astype(int)
    if "InternetService" in df_feat.columns:
        df_feat["total_services"] += (df_feat["InternetService"] != "No").astype(int)

    # 2. Security & Support Flags
    if "OnlineSecurity" in df_feat.columns:
        df_feat["has_security_service"] = (df_feat["OnlineSecurity"] == "Yes").astype(int)
    if "TechSupport" in df_feat.columns:
        df_feat["has_support_service"] = (df_feat["TechSupport"] == "Yes").astype(int)

    # 3. Contract & Payment Risk Indicators
    if "Contract" in df_feat.columns:
        df_feat["contract_month_to_month"] = (df_feat["Contract"] == "Month-to-month").astype(int)
    if "PaymentMethod" in df_feat.columns:
        df_feat["payment_electronic_check"] = (df_feat["PaymentMethod"] == "Electronic check").astype(int)

    # 4. Financial & Charge Ratios
    # Average monthly charge over tenure vs current monthly charge ratio
    expected_total = df_feat["MonthlyCharges"] * df_feat["tenure"]
    df_feat["charge_ratio"] = np.where(
        expected_total > 0,
        df_feat["TotalCharges"] / (expected_total + 1e-5),
        1.0
    )
    df_feat["charge_per_service"] = df_feat["MonthlyCharges"] / (df_feat["total_services"] + 1)

    logger.info(f"Feature engineering complete. Added new derived features. Total features: {df_feat.shape[1]}")
    return df_feat


def leakage_audit(df: pd.DataFrame) -> List[str]:
    """Audits columns to detect target leakage or identifier columns.

    Args:
        df: Feature DataFrame.

    Returns:
        List[str]: List of column names excluded from features.
    """
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]

    forbidden = [id_col, target_col]
    excluded_cols = [c for c in forbidden if c in df.columns]

    logger.info(f"Target Leakage Audit: Identified {len(excluded_cols)} excluded non-predictive/target columns: {excluded_cols}")
    return excluded_cols


def create_stratified_split(
    df: pd.DataFrame,
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Splits dataset into Stratified Train (70%), Validation (15%), and Test (15%) sets.

    Args:
        df: Fully engineered DataFrame.
        output_dir: Target directory relative to project root.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]: Train, Val, Test DataFrames and split metadata.
    """
    cfg = load_config()
    target_col = cfg["data"]["target_column"]
    random_state = cfg["project"]["random_state"]

    train_ratio = cfg["split"]["train_ratio"]  # 0.70
    val_ratio = cfg["split"]["val_ratio"]      # 0.15
    test_ratio = cfg["split"]["test_ratio"]    # 0.15

    val_test_ratio = val_ratio + test_ratio  # 0.30
    test_relative_ratio = test_ratio / val_test_ratio  # 0.50 of remaining 30%

    # Train / (Val + Test) split
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df[target_col]
    )

    # Val / Test split
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_relative_ratio,
        random_state=random_state,
        stratify=val_test_df[target_col]
    )

    metadata = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_churn_rate": float(train_df[target_col].mean()),
        "val_churn_rate": float(val_df[target_col].mean()),
        "test_churn_rate": float(test_df[target_col].mean())
    }

    # Save to disk
    root = get_project_root()
    proc_dir = root / output_dir
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    logger.info(f"Stratified Split Complete: Train={len(train_df)} ({metadata['train_churn_rate']:.2%}), Val={len(val_df)} ({metadata['val_churn_rate']:.2%}), Test={len(test_df)} ({metadata['test_churn_rate']:.2%})")

    return train_df, val_df, test_df, metadata


if __name__ == "__main__":
    raw_df, _ = ingest_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    feat_df = add_engineered_features(clean_df)
    leakage_audit(feat_df)
    train_df, val_df, test_df, meta = create_stratified_split(feat_df)
