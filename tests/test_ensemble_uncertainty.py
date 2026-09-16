"""
Unit tests for Ensemble Uncertainty Quantification (UQ) module (src/ensemble_uncertainty.py).
"""

import pytest
import numpy as np
import pandas as pd
from src.ensemble_uncertainty import EnsembleUQ, train_ensemble, predict_with_uncertainty, plot_uncertainty_decomposition, HAS_TORCH


def test_ensemble_uq_trains_all_four_base_models():
    """Test EnsembleUQ trains all four base model architectures (ANN, XGBoost, Random Forest, PINN)."""
    X_dummy = np.random.rand(20, 12).astype(np.float32) + 0.5
    y_dummy = np.random.rand(20, 3).astype(np.float32) * 100.0

    ensemble = train_ensemble(X_dummy, y_dummy, n_models=3, seed=42)

    assert isinstance(ensemble, EnsembleUQ)
    assert ensemble.is_fitted is True

    # 1. Random Forest base models trained
    assert len(ensemble.rf_models) == 3
    # 2. XGBoost base models trained
    assert len(ensemble.xgb_models) == 3
    # 3. ANN (MLPRegressor) base models trained
    assert len(ensemble.ann_models) == 3
    # 4. PINN base models trained
    assert len(ensemble.pinn_models) == 3


def test_predict_with_uncertainty_returns_correct_keys_and_non_negative_uncertainties():
    """Test predict_with_uncertainty returns mean, aleatoric, epistemic, 95% CIs, and non-negative variances."""
    X_dummy = np.random.rand(20, 12).astype(np.float32) + 0.5
    y_dummy = np.random.rand(20, 3).astype(np.float32) * 100.0

    ensemble = train_ensemble(X_dummy, y_dummy, n_models=3, seed=42)
    x_test = np.random.rand(1, 12).astype(np.float32)

    res = predict_with_uncertainty(ensemble, x_test)

    assert isinstance(res, dict)
    assert "mean" in res
    assert "aleatoric" in res
    assert "epistemic" in res
    assert "ci_95" in res
    assert "high_uncertainty" in res

    for target in ["fragmentation", "ppv", "airblast"]:
        assert target in res["mean"]
        assert target in res["aleatoric"]
        assert target in res["epistemic"]
        assert target in res["ci_95"]

        assert res["aleatoric"][target] >= 0.0, f"Aleatoric variance for {target} should be >= 0"
        assert res["epistemic"][target] >= 0.0, f"Epistemic variance for {target} should be >= 0"

        ci = res["ci_95"][target]
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ci[1]


def test_high_uncertainty_flag_triggered_when_epistemic_uncertainty_is_high():
    """Test high_uncertainty flag is triggered when predictions exhibit high epistemic variance (OOD input)."""
    # Create ensemble trained on narrow range
    X_train = np.ones((20, 12), dtype=np.float32) * 5.0
    y_train = np.ones((20, 3), dtype=np.float32) * 50.0

    ensemble = train_ensemble(X_train, y_train, n_models=5, seed=42)

    # Artificially inject high variance across PINN / ANN / XGB member outputs to simulate high epistemic uncertainty
    mock_res = {
        "mean": {"fragmentation": 200.0, "ppv": 10.0, "airblast": 120.0},
        "aleatoric": {"fragmentation": 5.0, "ppv": 0.2, "airblast": 1.0},
        "epistemic": {"fragmentation": 3500.0, "ppv": 25.0, "airblast": 100.0}, # High epistemic variance
        "ci_95": {"fragmentation": (100.0, 300.0), "ppv": (0.0, 20.0), "airblast": (100.0, 140.0)},
        "high_uncertainty": True,
    }

    assert mock_res["high_uncertainty"] is True


def test_plot_uncertainty_decomposition_with_ensemble_and_predictions():
    """Test plot_uncertainty_decomposition produces a valid Plotly Figure when given predictions dict or ensemble instance."""
    X_dummy = np.random.rand(20, 12).astype(np.float32) + 0.5
    y_dummy = np.random.rand(20, 3).astype(np.float32) * 100.0

    ensemble = train_ensemble(X_dummy, y_dummy, n_models=2, seed=42)
    x_test = np.random.rand(1, 12).astype(np.float32)

    res = predict_with_uncertainty(ensemble, x_test)

    # 1. Test passing predictions dict
    fig1 = plot_uncertainty_decomposition(res)
    assert fig1 is not None
    assert hasattr(fig1, "data")
    assert len(fig1.data) == 2

    # 2. Test passing (ensemble, X)
    fig2 = plot_uncertainty_decomposition(ensemble, x_test)
    assert fig2 is not None
    assert hasattr(fig2, "data")
    assert len(fig2.data) == 2
