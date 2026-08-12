"""
Data Cleaning Module for ChurnGuard AI.
Handles whitespace-to-numeric conversions, missing value imputations, target encoding, and data cleaning.
"""
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("DataCleaning")


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans raw Telco DataFrame and documents all transformations.

    Args:
        df: Raw input DataFrame.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and transformation metadata.
    """
    cleaned_df = df.copy()
    cfg = load_config()
    target_col = cfg["data"]["target_column"]

    metadata: Dict[str, Any] = {
        "initial_rows": len(df),
        "total_charges_imputed": 0,
        "target_distribution": {}
    }

    # 1. Strip column whitespace
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # 2. Fix TotalCharges whitespace & string conversion
    if "TotalCharges" in cleaned_df.columns:
        # Convert empty strings/spaces to NaN
        cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].astype(str).str.strip()
        cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")

        ws_mask = cleaned_df["TotalCharges"].isna()
        ws_count = int(ws_mask.sum())
        metadata["total_charges_imputed"] = ws_count

        if ws_count > 0:
            logger.info(f"Imputing {ws_count} missing TotalCharges values...")
            # For tenure == 0, TotalCharges is logically 0.0
            zero_tenure_mask = ws_mask & (cleaned_df["tenure"] == 0)
            cleaned_df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

            # Remaining NaNs (if any) filled with median TotalCharges
            remaining_nan = cleaned_df["TotalCharges"].isna()
            if remaining_nan.sum() > 0:
                median_tc = cleaned_df["TotalCharges"].median()
                cleaned_df.loc[remaining_nan, "TotalCharges"] = median_tc

    # 3. Standardize Target Column (Yes -> 1, No -> 0)
    if target_col in cleaned_df.columns:
        if cleaned_df[target_col].dtype == object or isinstance(cleaned_df[target_col].iloc[0], str):
            mapping = {"Yes": 1, "No": 0, "1": 1, "0": 0}
            cleaned_df[target_col] = cleaned_df[target_col].map(mapping).astype(int)

        target_counts = cleaned_df[target_col].value_counts().to_dict()
        metadata["target_distribution"] = {str(k): int(v) for k, v in target_counts.items()}
        churn_rate = cleaned_df[target_col].mean() * 100
        logger.info(f"Target distribution: {target_counts} | Churn Rate: {churn_rate:.2f}%")

    # 4. Remove duplicate rows if any
    dup_count = int(cleaned_df.duplicated().sum())
    if dup_count > 0:
        logger.info(f"Removing {dup_count} duplicate rows.")
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    metadata["final_rows"] = len(cleaned_df)

    logger.info(f"Data cleaning completed. Rows: {metadata['final_rows']}")
    return cleaned_df, metadata


def save_cleaned_data(cleaned_df: pd.DataFrame, output_path: str = "data/processed/cleaned_telco.csv") -> Path:
    """Saves cleaned DataFrame to designated CSV path.

    Args:
        cleaned_df: Cleaned DataFrame.
        output_path: Destination path relative to root.

    Returns:
        Path: Path where file was written.
    """
    root = get_project_root()
    full_path = root / output_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(full_path, index=False)
    logger.info(f"Cleaned dataset saved to: {full_path}")
    return full_path


if __name__ == "__main__":
    from .ingestion import ingest_raw_data
    raw_df, _ = ingest_raw_data()
    clean_df, meta = clean_dataset(raw_df)
    save_cleaned_data(clean_df)
