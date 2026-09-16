"""
Unit tests for synthetic data generation and data ingestion pipelines.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.synthetic_data import generate_synthetic_blast_data
from src.data_ingestion import (
    load_raw_data,
    load_real_blast_data,
    validate_data_schema,
    clean_and_preprocess,
    engineer_features,
    prepare_ingested_dataset,
    REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)


def test_synthetic_data_generation():
    df = generate_synthetic_blast_data(num_samples=50, seed=123)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    for target in TARGET_COLUMNS:
        assert target in df.columns
    assert (df["d50_mm"] > 0).all()
    assert (df["ppv_mms"] > 0).all()
    assert (df["flyrock_m"] > 0).all()


def test_data_ingestion_cleaning_and_features():
    df = generate_synthetic_blast_data(num_samples=30, seed=42)
    # Introduce duplicate and out-of-bounds value
    df.loc[0, "ppv_mms"] = -50.0  # Invalid non-physical value
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    is_valid, missing = validate_data_schema(df)
    assert is_valid
    assert len(missing) == 0

    cleaned = clean_and_preprocess(df)
    assert len(cleaned) == 30  # Duplicate dropped
    assert cleaned["ppv_mms"].min() >= 0.01  # Clipped to valid range

    engineered = engineer_features(cleaned)
    assert "spacing_burden_ratio" in engineered.columns
    assert "scaled_distance" in engineered.columns
    assert "energy_factor_mj_m3" in engineered.columns


def test_load_real_blast_data_and_anomaly_logging(tmp_path):
    """Test load_real_blast_data returns a valid DataFrame, handles missing values, and logs anomalies."""
    raw_df = generate_synthetic_blast_data(num_samples=20, seed=99)
    # Inject missing value and out-of-bounds entry
    raw_df.loc[2, "powder_factor_kg_m3"] = np.nan
    raw_df.loc[5, "burden_m"] = 99.0  # Out of range (max 15.0)

    csv_path = str(tmp_path / "test_real_blast_data.csv")
    log_path = str(tmp_path / "test_anomalies.log")
    raw_df.to_csv(csv_path, index=False)

    loaded_df = load_real_blast_data(csv_path, anomaly_log_path=log_path)

    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 20
    for col in REQUIRED_COLUMNS:
        assert col in loaded_df.columns
    assert not loaded_df["powder_factor_kg_m3"].isna().any(), "Expected missing values to be imputed"
    assert loaded_df["burden_m"].max() <= 15.0, "Expected out-of-bounds values to be clipped"

    # Verify log file was created and contains anomaly messages
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert "MISSING VALUE" in log_content
    assert "OUT OF RANGE" in log_content
