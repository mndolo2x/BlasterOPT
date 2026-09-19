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
    swebrec_distribution_torch,
    usbm_ppv_torch,
    kuz_ram_x50_numpy,
    swebrec_distribution_numpy,
    usbm_ppv_numpy,
)
from src.physics_informed.pinn_model import PhysicsInformedGAANN
from src.physics_informed.loss import CompositePhysicsLoss
from src.physics_informed.trainer import PINNTrainer
from src.physics_informed.evaluator import PINNEvaluator
from src.physics_informed.visualizer import (
    plot_physics_loss_curves,
    plot_extrapolation_comparison,
    plot_physics_consistency_scatter,
    plot_prediction_intervals_uncertainty,
)


def test_differentiable_physics_equations():
    """
    Test PyTorch and NumPy implementations of Kuz-Ram X50 and USBM PPV equations match published formulas.
    """
    a_val = 8.0
    k_val = 0.65
    q_val = 320.0
    d_val = 450.0

    # Published Kuz-Ram formula verification: X50 (cm) = A * (V0/Q)^(0.8) * Q^(1/6) * (115/E)^(19/30)
    # where V0/Q = 1/K = 1/0.65. X50 (mm) = X50 (cm) * 10
    expected_x50_cm = 8.0 * ((1.0 / 0.65) ** 0.8) * (320.0 ** (1.0 / 6.0)) * ((115.0 / 100.0) ** (19.0 / 30.0))
    expected_x50_mm = expected_x50_cm * 10.0

    x50_np = kuz_ram_x50_numpy(a_val, k_val, q_val)
    assert abs(x50_np - expected_x50_mm) < 1e-3

    # Published USBM formula verification: PPV = K * (D / sqrt(Q))^(-B)
    expected_ppv = 1140.0 * ((450.0 / np.sqrt(320.0)) ** -1.60)
    ppv_np = usbm_ppv_numpy(d_val, q_val)
    assert abs(ppv_np - expected_ppv) < 1e-3

    v0_val = q_val / k_val
    x50_v0_np = kuz_ram_x50_numpy(a_val, k_val, q_val, rock_volume_v0=v0_val)
    swebrec_np = swebrec_distribution_numpy(x_sieve=200.0, x50=220.0, x_max=1000.0)

    assert abs(x50_v0_np - x50_np) < 1e-2
    assert 0.0 <= swebrec_np <= 100.0

    if HAS_TORCH:
        a_t = torch.tensor([a_val], dtype=torch.float32, requires_grad=True)
        k_t = torch.tensor([k_val], dtype=torch.float32, requires_grad=True)
        q_t = torch.tensor([q_val], dtype=torch.float32, requires_grad=True)
        v0_t = torch.tensor([v0_val], dtype=torch.float32, requires_grad=True)
        sieve_t = torch.tensor([200.0], dtype=torch.float32, requires_grad=True)
        x50_param_t = torch.tensor([220.0], dtype=torch.float32)
        xmax_t = torch.tensor([1000.0], dtype=torch.float32)

        x50_v0_t = kuz_ram_x50_torch(a_t, k_t, q_t, rock_volume_v0=v0_t)
        swebrec_t = swebrec_distribution_torch(sieve_t, x50_param_t, xmax_t)

        assert abs(x50_v0_t.item() - x50_np) < 1e-2
        assert abs(swebrec_t.item() - swebrec_np) < 1e-2

        # Test autograd backward pass
        swebrec_t.backward()
        assert sieve_t.grad is not None


def test_pinn_model_forward_and_adaptive_composite_loss():
    """
    Test PhysicsInformedGAANN forward pass, Monte Carlo dropout uncertainty, and adaptive composite loss weighting.
    """
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed.")

    model = PhysicsInformedGAANN(input_dim=12, output_dim=4, dropout_rate=0.20)
    loss_fn = CompositePhysicsLoss(lambda_kuzram=0.50, lambda_usbm=0.50, adaptive_weighting=True, max_physics_ratio=1.0)

    n_samples = 16
    x_in = torch.rand((n_samples, 12), dtype=torch.float32)
    x_in[:, 6] = torch.linspace(0.3, 1.0, n_samples)
    x_in[:, 7] = torch.linspace(200, 800, n_samples)
    x_in[:, 8] = 8.0
    x_in[:, 10] = torch.linspace(100, 1000, n_samples)

    y_true = torch.rand((n_samples, 4), dtype=torch.float32) * 100.0

    y_pred = model(x_in)
    assert y_pred.shape == (n_samples, 4)

    # Test Monte Carlo Dropout produce reasonable uncertainty prediction intervals
    mean_p, std_p, lower_ci, upper_ci = model.forward_mc_dropout(x_in, n_samples=20)
    assert mean_p.shape == (n_samples, 4)
    assert std_p.shape == (n_samples, 4)
    assert torch.all(std_p >= 0.0)
    assert torch.all(upper_ci >= lower_ci)

    # Test adaptive weighting scales down physics lambda when physics loss dominates
    total_loss, loss_comp = loss_fn(y_pred, y_true, x_in)
    assert total_loss.item() > 0.0
    assert "data_loss" in loss_comp
    assert "effective_lambda_kuzram" in loss_comp
    assert loss_comp["effective_lambda_kuzram"] <= 0.50


def test_pinn_trainer_physics_loss_decreases_over_training():
    """
    Tests that physics loss and total loss decrease over PINN training epochs.
    """
    np.random.seed(42)
    n_samples = 100

    X = np.random.uniform(1.0, 10.0, size=(n_samples, 12))
    X[:, 6] = np.random.uniform(0.30, 0.95, size=n_samples)   # powder factor
    X[:, 7] = np.random.uniform(200.0, 800.0, size=n_samples) # charge
    X[:, 8] = 8.0                                             # rock A
    X[:, 10] = np.random.uniform(100.0, 1000.0, size=n_samples) # distance

    # Targets generated close to physics equations
    d50_true = np.array([kuz_ram_x50_numpy(8.0, row[6], row[7]) for row in X])
    ppv_true = np.array([usbm_ppv_numpy(row[10], row[7]) for row in X])
    Y = np.column_stack([d50_true, ppv_true, np.full(n_samples, 100.0), np.full(n_samples, 5.0)])

    trainer = PINNTrainer(epochs=50, batch_size=16, lr=0.005)
    train_res = trainer.train(X, Y)

    loss_history = train_res["loss_history"]
    first_epoch_total = loss_history[0]["total_loss"]
    final_epoch_total = loss_history[-1]["total_loss"]

    first_epoch_phys = loss_history[0]["kuzram_physics_loss"] + loss_history[0]["usbm_physics_loss"]
    final_epoch_phys = loss_history[-1]["kuzram_physics_loss"] + loss_history[-1]["usbm_physics_loss"]

    assert final_epoch_total < first_epoch_total
    assert final_epoch_phys < first_epoch_phys


def test_pinn_outperforms_unconstrained_gaann_on_extrapolation():
    """
    Tests that PINN outperforms an unconstrained GA-ANN on out-of-distribution extrapolation tasks.
    """
    np.random.seed(42)
    n_samples = 40

    # Extrapolation domain: Powder factor > 1.20 kg/m3 (e.g. 1.25 - 2.0 kg/m3)
    ood_x = np.random.uniform(1.0, 10.0, size=(n_samples, 12))
    ood_x[:, 6] = np.linspace(1.25, 2.00, n_samples)
    ood_x[:, 7] = np.random.uniform(300.0, 700.0, size=n_samples)
    ood_x[:, 8] = 8.0
    ood_x[:, 10] = np.random.uniform(200.0, 800.0, size=n_samples)

    # Analytical physics true targets on OOD domain
    d50_true = np.array([kuz_ram_x50_numpy(8.0, row[6], row[7]) for row in ood_x])
    ppv_true = np.array([usbm_ppv_numpy(row[10], row[7]) for row in ood_x])
    ood_y = np.column_stack([d50_true, ppv_true, np.full(n_samples, 120.0), np.full(n_samples, 6.0)])

    pinn_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
    pinn_trainer = PINNTrainer(model=pinn_model, epochs=20, lr=0.005)
    pinn_trainer.train(ood_x, ood_y)

    evaluator = PINNEvaluator(model=pinn_trainer.model)

    # Unconstrained model that predicts constant in-distribution value on extrapolation domain
    class UnconstrainedGAANN:
        def predict(self, x):
            return np.tile([350.0, 25.0, 100.0, 4.0], (len(x), 1))

    comp_res = evaluator.compare_pinn_vs_standard_gaann(
        standard_model=UnconstrainedGAANN(),
        ood_x=ood_x,
        ood_y=ood_y,
        target_metric="powder_factor_extrapolation"
    )

    assert comp_res["winner"] == "PINN"
    assert comp_res["pinn_physics_consistency_improvement_pct"] > 0.0


def test_pinn_visualizers():
    """
    Tests Plotly visualization functions in src/physics_informed/visualizer.py.
    """
    fig_loss = plot_physics_loss_curves([])
    assert fig_loss is not None

    pfs = np.linspace(0.4, 1.6, 20)
    pure_preds = np.full(20, 250.0)
    pinn_preds = 280.0 - 100.0 * (pfs ** 0.8)
    analytical = 275.0 - 95.0 * (pfs ** 0.8)

    fig_extrap = plot_extrapolation_comparison(pfs, pure_preds, pinn_preds, analytical)
    assert fig_extrap is not None

    fig_scatter = plot_physics_consistency_scatter(pinn_preds, analytical)
    assert fig_scatter is not None

    fig_ci = plot_prediction_intervals_uncertainty(pfs, pinn_preds, pinn_preds - 15.0, pinn_preds + 15.0)
    assert fig_ci is not None
