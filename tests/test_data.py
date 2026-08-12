"""
Unit tests for data ingestion, download, validation, and cleaning modules.
"""
from pathlib import Path
import pandas as pd
import pytest
from src.data.cleaning import clean_dataset, save_cleaned_data
from src.data.download import download_raw_dataset
from src.data.ingestion import ingest_raw_data
from src.data.validation import validate_raw_data, DataValidationError
from src.utils.config import get_project_root, load_config


def test_project_root_exists():
    """Verify that project root directory is identified correctly."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "config.yaml").exists()


def test_load_config():
    """Verify that config.yaml loads valid YAML settings."""
    cfg = load_config()
    assert "project" in cfg
    assert cfg["project"]["name"] == "ChurnGuard AI"
    assert "data" in cfg
    assert cfg["data"]["target_column"] == "Churn"


def test_download_and_ingestion():
    """Verify that raw dataset is ingested correctly."""
    df, metadata = ingest_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert metadata["n_rows"] > 5000
    assert metadata["n_cols"] == 21
    assert "Churn" in df.columns


def test_validate_raw_data_success():
    """Verify that valid dataset passes automated data validation."""
    df, _ = ingest_raw_data()
    report = validate_raw_data(df)
    assert report["status"] == "PASSED"
    assert report["missing_columns"] == []
    assert report["total_rows"] == len(df)


def test_validate_raw_data_missing_column():
    """Verify that validation raises DataValidationError on missing required columns."""
    df, _ = ingest_raw_data()
    df_incomplete = df.drop(columns=["Churn"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(df_incomplete)
    assert "Missing required columns" in str(exc_info.value) or "Target column 'Churn' not found" in str(exc_info.value)


def test_clean_dataset_imputation_and_encoding():
    """Verify that clean_dataset correctly handles whitespace TotalCharges and binary Churn mapping."""
    raw_df, _ = ingest_raw_data()
    clean_df, meta = clean_dataset(raw_df)

    assert meta["final_rows"] > 0
    # Verify TotalCharges is numeric float without NaNs
    assert pd.api.types.is_float_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].isna().sum() == 0

    # Verify Churn is binary (0 or 1)
    assert set(clean_df["Churn"].unique()).issubset({0, 1})
    assert clean_df["Churn"].dtype == int


def test_save_cleaned_data(tmp_path):
    """Verify saving cleaned data to CSV file."""
    raw_df, _ = ingest_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    out_file = tmp_path / "cleaned.csv"
    saved_path = save_cleaned_data(clean_df, str(out_file))
    assert saved_path.exists()
    reloaded = pd.read_csv(saved_path)
    assert len(reloaded) == len(clean_df)
