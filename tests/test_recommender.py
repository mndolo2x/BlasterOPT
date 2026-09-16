"""
Unit tests for the Similar Blast Recommender module.
"""

import pytest
import pandas as pd
import numpy as np
from src.recommender import find_similar_blasts
from src.synthetic_data import generate_synthetic_blast_data


def test_find_similar_blasts_returns_dataframe_and_correct_rows():
    """Test that find_similar_blasts returns a DataFrame with top_k rows."""
    df_hist = generate_synthetic_blast_data(num_samples=50, seed=42)
    new_params = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }

    res = find_similar_blasts(new_params, df_hist, top_k=5)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 5
    assert "similarity_distance" in res.columns
    assert "d50_mm" in res.columns
    assert "ppv_mms" in res.columns


def test_find_similar_blasts_non_negative_distances():
    """Test that similarity distances in returned DataFrame are all non-negative."""
    df_hist = generate_synthetic_blast_data(num_samples=30, seed=100)
    new_params = {
        "burden_m": 5.5,
        "spacing_m": 6.5,
        "stemming_m": 4.5,
        "powder_factor_kg_m3": 0.60,
    }

    res = find_similar_blasts(new_params, df_hist, top_k=3)

    assert len(res) == 3
    distances = res["similarity_distance"].values
    assert np.all(distances >= 0.0), f"Expected non-negative distances, got: {distances}"


def test_find_similar_blasts_empty_history():
    """Test that find_similar_blasts handles empty historical DataFrame gracefully."""
    empty_df = pd.DataFrame()
    new_params = {"burden_m": 6.0, "spacing_m": 7.0}

    res = find_similar_blasts(new_params, empty_df, top_k=5)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
