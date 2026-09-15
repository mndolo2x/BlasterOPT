import pytest
import pandas as pd
import numpy as np
from src.synthetic_data import generate_synthetic_data, generate_synthetic_blast_data


def test_generate_synthetic_data_rows_and_columns():
    """Test that generate_synthetic_data(n_samples=100) returns a DataFrame with correct rows and columns."""
    df = generate_synthetic_data(n_samples=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100

    expected_cols = [
        "blast_id",
        "mine_site",
        "rock_type",
        "rock_density_t_m3",
        "rock_factor_A",
        "explosive_type",
        "explosive_rws",
        "explosive_density_g_cm3",
        "bench_height_m",
        "hole_diameter_mm",
        "hole_depth_m",
        "burden_m",
        "spacing_m",
        "stemming_m",
        "subdrilling_m",
        "charge_mass_per_hole_kg",
        "powder_factor_kg_m3",
        "powder_factor_kg_t",
        "max_charge_per_delay_kg",
        "monitoring_distance_m",
        "d50_mm",
        "uniformity_index_n",
        "ppv_mms",
        "flyrock_m",
        "cost_per_tonne_usd",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_input_parameter_ranges():
    """Test that generated input parameters fall within specified physical ranges."""
    df = generate_synthetic_data(n_samples=100, rock_factor_range=(6.0, 12.0), bench_height_range=(10.0, 15.0))

    assert (df["rock_factor_A"] >= 6.0).all() and (df["rock_factor_A"] <= 12.0).all()
    assert (df["bench_height_m"] >= 10.0).all() and (df["bench_height_m"] <= 15.0).all()
    assert (df["hole_diameter_mm"] >= 150.0).all() and (df["hole_diameter_mm"] <= 311.0).all()
    assert (df["rock_density_t_m3"] >= 2.0).all() and (df["rock_density_t_m3"] <= 3.2).all()
    assert (df["burden_m"] > 0).all()
    assert (df["spacing_m"] > 0).all()
    assert (df["stemming_m"] > 0).all()
    assert (df["monitoring_distance_m"] >= 150.0).all() and (df["monitoring_distance_m"] <= 1200.0).all()


def test_output_targets_positive():
    """Test that all output targets are strictly positive numbers."""
    df = generate_synthetic_data(n_samples=100)

    target_cols = ["d50_mm", "uniformity_index_n", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]
    for col in target_cols:
        assert (df[col] > 0).all(), f"Target column {col} contains non-positive values"
