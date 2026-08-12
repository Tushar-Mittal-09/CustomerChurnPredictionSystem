"""
Automated Data Validation Pipeline for ChurnGuard AI.
Inspects data integrity, schema consistency, required columns, and value ranges.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger("DataValidation")

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


class DataValidationError(Exception):
    """Custom exception raised when automated data validation fails."""
    pass


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates raw DataFrame schema, missing values, column ranges, and target integrity.

    Args:
        df: Pandas DataFrame containing raw Telco data.

    Returns:
        Dict[str, Any]: Detailed validation report dictionary.

    Raises:
        DataValidationError: If critical checks fail.
    """
    cfg = load_config()
    target_col = cfg["data"]["target_column"]

    report: Dict[str, Any] = {
        "status": "PASSED",
        "missing_columns": [],
        "invalid_target_values": [],
        "whitespace_total_charges_count": 0,
        "negative_charges_count": 0,
        "duplicate_customer_ids": 0,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "warnings": []
    }

    # 1. Check Required Columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["status"] = "FAILED"
        raise DataValidationError(f"Missing required columns in dataset: {missing_cols}")

    # 2. Check Target Column
    if target_col not in df.columns:
        report["status"] = "FAILED"
        raise DataValidationError(f"Target column '{target_col}' not found in dataset.")

    unique_target = df[target_col].dropna().unique().tolist()
    valid_target_sets = [{"Yes", "No"}, {0, 1}, {"0", "1"}, {"Yes", "No", np.nan}]
    if not any(set(unique_target).issubset(s) for s in valid_target_sets):
        report["invalid_target_values"] = unique_target
        report["status"] = "FAILED"
        raise DataValidationError(f"Target column '{target_col}' contains invalid values: {unique_target}")

    # 3. Check Customer ID Uniqueness
    if "customerID" in df.columns:
        duplicate_ids = df["customerID"].duplicated().sum()
        report["duplicate_customer_ids"] = int(duplicate_ids)
        if duplicate_ids > 0:
            report["warnings"].append(f"Found {duplicate_ids} duplicate customer IDs.")

    # 4. Check Whitespace Strings in TotalCharges
    if "TotalCharges" in df.columns:
        whitespace_mask = df["TotalCharges"].astype(str).str.strip() == ""
        ws_count = int(whitespace_mask.sum())
        report["whitespace_total_charges_count"] = ws_count
        if ws_count > 0:
            report["warnings"].append(f"Found {ws_count} rows with blank/whitespace TotalCharges.")

    # 5. Check Non-Negative Charges & Tenure
    for num_col in ["tenure", "MonthlyCharges"]:
        if num_col in df.columns:
            converted = pd.to_numeric(df[num_col], errors="coerce")
            neg_count = int((converted < 0).sum())
            if neg_count > 0:
                report["negative_charges_count"] += neg_count
                report["warnings"].append(f"Found {neg_count} negative values in '{num_col}'.")

    logger.info(f"Data validation passed successfully with {len(report['warnings'])} warnings.")
    return report


if __name__ == "__main__":
    from .ingestion import ingest_raw_data
    raw_df, _ = ingest_raw_data()
    val_report = validate_raw_data(raw_df)
    print("Validation Report:", val_report)
