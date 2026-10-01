"""
Unit tests for AirblastMinimizerModel in src/models.py.
"""
import pytest
import numpy as np
import pandas as pd
from src.models import AirblastMinimizerModel
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features


def test_airblast_predicts_one_output():
    assert AirblastMinimizerModel.OUTPUT_COLUMNS == ["airblast_db"]


def test_stemming_is_most_sensitive():
    df = engineer_features(generate_synthetic_data(n_samples=50, seed=42))
    m = AirblastMinimizerModel(input_size=8)
    m.fit(df, df[AirblastMinimizerModel.OUTPUT_COLUMNS])

    sens = m.compute_sensitivity(df.head(1))
    top_3_features = [item[0] for item in sens[:3]]

    assert "stemming_m" in top_3_features, f"Expected stemming_m in top 3 sensitive features, got {top_3_features}"


def test_spacing_is_least_sensitive():
    df = engineer_features(generate_synthetic_data(n_samples=50, seed=42))
    m = AirblastMinimizerModel(input_size=8)
    m.fit(df, df[AirblastMinimizerModel.OUTPUT_COLUMNS])

    sens = m.compute_sensitivity(df.head(1))
    bottom_3_features = [item[0] for item in sens[-3:]]

    assert "spacing_m" in bottom_3_features, f"Expected spacing_m in bottom 3 sensitive features, got {bottom_3_features}"


def test_minimization_reduces_airblast():
    df = engineer_features(generate_synthetic_data(n_samples=50, seed=42))
    m = AirblastMinimizerModel(input_size=8)
    m.fit(df, df[AirblastMinimizerModel.OUTPUT_COLUMNS])

    sample = df.head(1)
    initial_airblast = m.predict(sample)["airblast_db"].values[0]

    min_df = m.minimize_airblast(sample)
    optimized_airblast = m.predict(min_df)["airblast_db"].values[0]

    assert optimized_airblast <= initial_airblast, f"Expected optimized airblast ({optimized_airblast}) <= initial ({initial_airblast})"
