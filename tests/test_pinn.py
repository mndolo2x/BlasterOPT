"""
Unit tests for Physics-Informed Neural Network (PINN) module (src/pinn.py).
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


def test_train_pinn_executes_successfully():
    """Test train_pinn executes training loop and returns model and loss history."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    pinn = BlastPINN(input_dim=12)
    X_dummy = np.random.rand(20, 12).astype(np.float32) + 0.5
    y_dummy = np.random.rand(20, 3).astype(np.float32) * 100.0

    model_trained, history = train_pinn(pinn, X_dummy, y_dummy, epochs=5, lr=0.001)

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) > 0


def test_predict_with_uncertainty_returns_exact_keys():
    """Test predict_with_uncertainty returns mean, std, ci_95, aleatoric, epistemic, and high_uncertainty."""
    if HAS_TORCH:
        pinn = BlastPINN(input_dim=12)
        x_sample = np.random.randn(1, 12).astype(np.float32)
        x_sample[0, 6] = 0.65   # powder factor
        x_sample[0, 7] = 640.0  # max charge
        x_sample[0, 10] = 450.0 # distance
        x_sample[0, 11] = 8.0   # blastability index / rock factor
    else:
        pinn = None
        x_sample = np.random.randn(1, 12)

    res = predict_with_uncertainty(pinn, x_sample, n_samples=30)

    assert isinstance(res, dict)
    assert "mean" in res
    assert "std" in res
    assert "ci_95" in res
    assert "aleatoric" in res
    assert "epistemic" in res
    assert "high_uncertainty" in res

    for key in ["fragmentation", "ppv", "airblast"]:
        assert key in res["mean"]
        assert key in res["std"]
        assert key in res["ci_95"]
        assert key in res["aleatoric"]
        assert key in res["epistemic"]

    ci_frag = res["ci_95"]["fragmentation"]
    assert isinstance(ci_frag, tuple)
    assert len(ci_frag) == 2
    assert ci_frag[0] <= ci_frag[1]
