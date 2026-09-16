"""
Unit tests for Physics-Informed Neural Network (PINN) module.
"""

import pytest
import numpy as np
from src.pinn import BlastPINN, train_pinn, predict_with_uncertainty, HAS_TORCH

if HAS_TORCH:
    import torch


def test_blast_pinn_forward_pass_returns_three_outputs():
    """Test BlastPINN forward pass returns three output tensors (frag, ppv, airblast)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    pinn = BlastPINN(input_dim=12)
    x = torch.randn(8, 12)

    pred_frag, pred_ppv, pred_air = pinn(x)

    assert pred_frag.shape == (8, 1)
    assert pred_ppv.shape == (8, 1)
    assert pred_air.shape == (8, 1)


def test_predict_with_uncertainty_returns_mean_and_confidence_intervals():
    """Test predict_with_uncertainty returns mean, std, 95% CI, and OOD flag."""
    if HAS_TORCH:
        pinn = BlastPINN(input_dim=12)
        x_sample = np.random.randn(1, 12)
        # Ensure positive inputs for physics loss equations
        x_sample[0, 6] = 0.65  # powder factor
        x_sample[0, 7] = 640.0 # max charge
        x_sample[0, 10] = 450.0 # distance
    else:
        pinn = None
        x_sample = np.random.randn(1, 12)

    res = predict_with_uncertainty(pinn, x_sample, n_samples=30)

    assert isinstance(res, dict)
    assert "mean" in res
    assert "std" in res
    assert "confidence_interval_95" in res
    assert "is_out_of_distribution" in res

    assert "fragmentation_d50_mm" in res["mean"]
    assert "ppv_mms" in res["mean"]
    assert "airblast_dbl" in res["mean"]

    ci_frag = res["confidence_interval_95"]["fragmentation_d50_mm"]
    assert isinstance(ci_frag, tuple)
    assert len(ci_frag) == 2
    assert ci_frag[0] <= ci_frag[1]
