"""
Unit tests for Ensemble Uncertainty Quantification (UQ) module (src/ensemble_uncertainty.py).
"""

import pytest
import numpy as np
import pandas as pd
from src.ensemble_uncertainty import EnsembleUQ, train_ensemble, predict_with_uncertainty, plot_uncertainty_decomposition


def test_train_ensemble_returns_fitted_ensemble():
    """Test train_ensemble fits ensemble members across 4 base architectures."""
    X_dummy = np.random.rand(20, 12).astype(np.float32) + 0.5
    y_dummy = np.random.rand(20, 3).astype(np.float32) * 100.0

    ensemble = train_ensemble(X_dummy, y_dummy, n_models=3, seed=42)

    assert isinstance(ensemble, EnsembleUQ)
    assert ensemble.is_fitted is True
    assert len(ensemble.rf_models) == 3
    assert len(ensemble.xgb_models) == 3
    assert len(ensemble.ann_models) == 3


def test_predict_with_uncertainty_returns_non_negative_uncertainties():
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


def test_plot_uncertainty_decomposition_returns_figure():
    """Test plot_uncertainty_decomposition produces a valid Plotly Figure."""
    X_dummy = np.random.rand(20, 12).astype(np.float32) + 0.5
    y_dummy = np.random.rand(20, 3).astype(np.float32) * 100.0

    ensemble = train_ensemble(X_dummy, y_dummy, n_models=2, seed=42)
    x_test = np.random.rand(1, 12).astype(np.float32)

    res = predict_with_uncertainty(ensemble, x_test)
    fig = plot_uncertainty_decomposition(res)

    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) == 2  # Aleatoric + Epistemic bars
