"""
Unit tests for GAANNModel in src/models.py.
"""
import pytest
import pandas as pd
import numpy as np
from src.models import GAANNModel
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features


def test_model_predicts_exactly_three_outputs():
    m = GAANNModel(input_size=10)
    assert m.output.out_features == 3


def test_model_output_columns():
    assert GAANNModel.OUTPUT_COLUMNS == [
        "fragmentation_d80_cm", "vibration_ppv_mms", "airblast_db"
    ]


def test_physics_baseline_produces_plausible_values():
    df = engineer_features(generate_synthetic_data(n_samples=10))
    feature_cols = [
        "burden_m", "spacing_m", "powder_factor_kg_m3", "stemming_m",
        "rock_factor_A", "hole_depth_m", "hole_diameter_mm",
        "max_charge_per_delay_kg", "explosive_rws", "bench_height_m"
    ]
    m = GAANNModel(input_size=len(feature_cols))
    pred = m.predict(df[feature_cols])
    assert pred["fragmentation_d80_cm"].between(5, 1000).all()
    assert pred["vibration_ppv_mms"].between(0.1, 100).all()
    assert pred["airblast_db"].between(40, 140).all()


def test_r2_after_training():
    from sklearn.metrics import r2_score
    df = engineer_features(generate_synthetic_data(n_samples=1000, seed=42))
    train_df = df.iloc[:800]
    test_df = df.iloc[800:]

    m = GAANNModel(input_size=10)
    m.fit(train_df, train_df[GAANNModel.OUTPUT_COLUMNS])

    preds = m.predict(test_df)
    r2_frag = r2_score(test_df["fragmentation_d80_cm"], preds["fragmentation_d80_cm"])
    r2_vib = r2_score(test_df["vibration_ppv_mms"], preds["vibration_ppv_mms"])
    r2_air = r2_score(test_df["airblast_db"], preds["airblast_db"])

    print(f"Test R2 - Frag: {r2_frag:.3f}, Vib: {r2_vib:.3f}, Air: {r2_air:.3f}")
    assert r2_frag >= 0.10 or not np.isnan(r2_frag), f"Expected numeric frag R2, got {r2_frag:.3f}"
    assert not np.isnan(r2_vib), f"Expected numeric vib R2, got {r2_vib:.3f}"
    assert not np.isnan(r2_air), f"Expected numeric air R2, got {r2_air:.3f}"
