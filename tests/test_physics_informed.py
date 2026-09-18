"""
Unit tests for Physics-Informed Neural Network Package (`src/physics_informed`).
"""

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.physics_informed.physics_equations import (
    kuz_ram_x50_torch,
    usbm_ppv_torch,
    kuz_ram_x50_numpy,
    usbm_ppv_numpy,
)
from src.physics_informed.pinn_model import PhysicsInformedGAANN
from src.physics_informed.loss import CompositePhysicsLoss
from src.physics_informed.trainer import PINNTrainer
from src.physics_informed.evaluator import PINNEvaluator


def test_differentiable_physics_equations():
    """
    Test PyTorch and NumPy implementations of Kuz-Ram X50 and USBM PPV equations.
    """
    a_val = 8.0
    k_val = 0.65
    q_val = 320.0
    d_val = 450.0

    # NumPy calculations
    x50_np = kuz_ram_x50_numpy(a_val, k_val, q_val)
    ppv_np = usbm_ppv_numpy(d_val, q_val)

    assert x50_np > 0.0
    assert ppv_np > 0.0

    if HAS_TORCH:
        a_t = torch.tensor([a_val], dtype=torch.float32, requires_grad=True)
        k_t = torch.tensor([k_val], dtype=torch.float32, requires_grad=True)
        q_t = torch.tensor([q_val], dtype=torch.float32, requires_grad=True)
        d_t = torch.tensor([d_val], dtype=torch.float32, requires_grad=True)

        x50_t = kuz_ram_x50_torch(a_t, k_t, q_t)
        ppv_t = usbm_ppv_torch(d_t, q_t)

        assert abs(x50_t.item() - x50_np) < 1e-3
        assert abs(ppv_t.item() - ppv_np) < 1e-3

        # Test autograd backward pass
        x50_t.backward()
        assert a_t.grad is not None
        assert k_t.grad is not None


def test_pinn_model_forward_and_composite_loss():
    """
    Test PhysicsInformedGAANN forward pass and CompositePhysicsLoss evaluation.
    """
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed.")

    model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
    loss_fn = CompositePhysicsLoss(lambda_kuzram=0.25, lambda_usbm=0.25)

    n_samples = 16
    x_in = torch.rand((n_samples, 12), dtype=torch.float32)
    # Set valid realistic scaling for physics features
    x_in[:, 6] = torch.linspace(0.3, 1.0, n_samples)   # powder factor
    x_in[:, 7] = torch.linspace(200, 800, n_samples)   # charge per delay
    x_in[:, 8] = 8.0                                   # rock factor A
    x_in[:, 10] = torch.linspace(100, 1000, n_samples) # distance

    y_true = torch.rand((n_samples, 4), dtype=torch.float32) * 100.0

    y_pred = model(x_in)
    assert y_pred.shape == (n_samples, 4)

    total_loss, loss_comp = loss_fn(y_pred, y_true, x_in)
    assert total_loss.item() > 0.0
    assert "data_loss" in loss_comp
    assert "kuzram_physics_loss" in loss_comp
    assert "usbm_physics_loss" in loss_comp


def test_pinn_trainer_and_extrapolation():
    """
    Tests training the PINN and evaluating OOD extrapolation performance.
    """
    np.random.seed(42)
    n_samples = 80

    X = np.random.uniform(1.0, 10.0, size=(n_samples, 12))
    X[:, 6] = np.random.uniform(0.30, 0.95, size=n_samples)   # powder factor
    X[:, 7] = np.random.uniform(200.0, 800.0, size=n_samples) # charge
    X[:, 8] = 8.0                                             # rock A
    X[:, 10] = np.random.uniform(100.0, 1000.0, size=n_samples) # distance

    Y = np.random.uniform(10.0, 300.0, size=(n_samples, 4))

    trainer = PINNTrainer(epochs=10, batch_size=16)
    train_res = trainer.train(X, Y)
    assert train_res["epochs_completed"] == 10

    evaluator = PINNEvaluator(model=trainer.model)
    ood_x = X.copy()
    ood_x[:, 6] = np.linspace(1.25, 2.0, n_samples) # Out-of-range powder factor > 1.20

    extrap_res = evaluator.evaluate_extrapolation(ood_x, target_metric="powder_factor_kg_m3")
    assert "d50_physics_mae_mm" in extrap_res
    assert "extrapolation_status" in extrap_res
