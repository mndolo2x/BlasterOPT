"""
Unit tests for data source selector and training data verification.
"""
import pytest
import pandas as pd
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import dataframe_fingerprint
from src.models import GAANNModel, FEATURE_COLS


def test_fingerprint_changes_with_data():
    df_a = generate_synthetic_data(n_samples=100, seed=1)
    df_b = generate_synthetic_data(n_samples=100, seed=99)

    assert dataframe_fingerprint(df_a) != dataframe_fingerprint(df_b)


def test_fingerprint_is_deterministic():
    df_a = generate_synthetic_data(n_samples=100, seed=42)
    df_b = generate_synthetic_data(n_samples=100, seed=42)

    assert dataframe_fingerprint(df_a) == dataframe_fingerprint(df_b)


def test_training_uses_selected_dataframe():
    df_a = generate_synthetic_data(n_samples=100, seed=1)
    df_b = generate_synthetic_data(n_samples=100, seed=99)

    m_a = GAANNModel(input_size=len(FEATURE_COLS))
    m_a.fit(df_a[FEATURE_COLS], df_a[m_a.OUTPUT_COLUMNS])

    m_b = GAANNModel(input_size=len(FEATURE_COLS))
    m_b.fit(df_b[FEATURE_COLS], df_b[m_b.OUTPUT_COLUMNS])

    pred_a = m_a.predict(df_a[FEATURE_COLS].head(5))
    pred_b = m_b.predict(df_a[FEATURE_COLS].head(5))

    assert not pred_a.equals(pred_b), (
        "Two models trained on different data produced identical predictions. "
        "Training is not actually using the data."
    )


def test_model_learns_something():
    df = generate_synthetic_data(n_samples=500)
    m = GAANNModel(input_size=len(FEATURE_COLS))
    m.fit(df[FEATURE_COLS], df[m.OUTPUT_COLUMNS])

    preds = m.predict(df[FEATURE_COLS].head(50))
    for col in m.OUTPUT_COLUMNS:
        assert preds[col].std() > 1e-6, (
            f"Predictions for {col} have zero variance. Model did not learn."
        )
