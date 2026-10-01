"""
Unit tests for PSOANNModel in src/models.py.
"""
import pytest
import numpy as np
import pandas as pd
from src.models import PSOANNModel
from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import engineer_features


def test_pso_ann_predicts_one_output():
    assert PSOANNModel.OUTPUT_COLUMNS == ["fragmentation_d80_cm"]


def test_pso_ann_input_size_is_seven():
    m = PSOANNModel(input_size=7)
    assert m.hidden1.in_features == 7


def test_pso_training_reduces_loss():
    df = engineer_features(generate_synthetic_data(n_samples=50, seed=42))
    m = PSOANNModel(input_size=7)
    m.fit(df, df[PSOANNModel.OUTPUT_COLUMNS], n_particles=10, n_iterations=15)

    assert len(m.loss_history) == 16
    assert m.loss_history[-1] <= m.loss_history[0]


def test_r2_after_training():
    from sklearn.metrics import r2_score
    df = engineer_features(generate_synthetic_data(n_samples=500, seed=42))
    train_df = df.iloc[:400]
    test_df = df.iloc[400:]

    m = PSOANNModel(input_size=7)
    m.fit(train_df, train_df[PSOANNModel.OUTPUT_COLUMNS], n_particles=15, n_iterations=20)

    preds = m.predict(test_df)
    r2_frag = r2_score(test_df["fragmentation_d80_cm"], preds["fragmentation_d80_cm"])

    print(f"PSO-ANN Test R2 - Frag: {r2_frag:.3f}")
    assert not np.isnan(r2_frag)
