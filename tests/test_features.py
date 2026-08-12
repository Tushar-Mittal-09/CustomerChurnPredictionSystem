"""
Unit tests for feature engineering, leakage audit, and stratified splitting.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data.cleaning import clean_dataset
from src.data.ingestion import ingest_raw_data
from src.features.engineering import add_engineered_features, leakage_audit, create_stratified_split
from src.utils.config import get_project_root


def test_add_engineered_features():
    """Verify feature creation logic adds total_services, risk flags, and ratios."""
    raw_df, _ = ingest_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    feat_df = add_engineered_features(clean_df)

    assert "total_services" in feat_df.columns
    assert "has_security_service" in feat_df.columns
    assert "has_support_service" in feat_df.columns
    assert "charge_ratio" in feat_df.columns
    assert "charge_per_service" in feat_df.columns

    # Verify no infinite or NaN values in engineered features
    assert not np.isinf(feat_df["charge_ratio"]).any()
    assert not np.isnan(feat_df["charge_ratio"]).any()
    assert not np.isnan(feat_df["total_services"]).any()


def test_leakage_audit():
    """Verify leakage audit accurately flags target and identifier columns."""
    raw_df, _ = ingest_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    feat_df = add_engineered_features(clean_df)

    excluded = leakage_audit(feat_df)
    assert "customerID" in excluded
    assert "Churn" in excluded


def test_create_stratified_split(tmp_path):
    """Verify stratified splitting maintains proportion and creates train, val, test files."""
    raw_df, _ = ingest_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    feat_df = add_engineered_features(clean_df)

    train_df, val_df, test_df, meta = create_stratified_split(feat_df, output_dir=str(tmp_path))

    assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
    # Check stratification: churn rates in splits should be close (~26.5%)
    overall_churn = feat_df["Churn"].mean()
    assert abs(meta["train_churn_rate"] - overall_churn) < 0.02
    assert abs(meta["val_churn_rate"] - overall_churn) < 0.02
    assert abs(meta["test_churn_rate"] - overall_churn) < 0.02

    # Check files saved
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()
