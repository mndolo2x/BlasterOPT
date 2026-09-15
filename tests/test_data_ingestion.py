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
