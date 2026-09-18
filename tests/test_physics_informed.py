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
    Test PyTorch and NumPy implementations of Kuz-Ram X50 and USBM PPV equations.
    """
    a_val = 8.0
    k_val = 0.65
    q_val = 320.0
    d_val = 450.0

    # NumPy calculations
    x50_np = kuz_ram_x50_numpy(a_val, k_val, q_val)
    ppv_np = usbm_ppv_numpy(d_val, q_val)
    v0_val = q_val / k_val
    x50_v0_np = kuz_ram_x50_numpy(a_val, k_val, q_val, rock_volume_v0=v0_val)
    swebrec_np = swebrec_distribution_numpy(x_sieve=200.0, x50=220.0, x_max=1000.0)

    assert x50_np > 0.0
    assert ppv_np > 0.0
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


def test_pinn_trainer_validation_early_stopping_and_curriculum():
    """
    Tests validation early stopping, CosineAnnealingLR, best model R2 saving, and curriculum mode in PINNTrainer.
    """
    np.random.seed(42)
    n_train = 60
    n_val = 20

    X_train = np.random.uniform(1.0, 10.0, size=(n_train, 12))
    X_train[:, 6] = np.random.uniform(0.30, 0.95, size=n_train)
    X_train[:, 7] = np.random.uniform(200.0, 800.0, size=n_train)
    X_train[:, 8] = 8.0
    X_train[:, 10] = np.random.uniform(100.0, 1000.0, size=n_train)
    Y_train = np.random.uniform(10.0, 300.0, size=(n_train, 4))

    X_val = np.random.uniform(1.0, 10.0, size=(n_val, 12))
    X_val[:, 6] = np.random.uniform(0.30, 0.95, size=n_val)
    X_val[:, 7] = np.random.uniform(200.0, 800.0, size=n_val)
    X_val[:, 8] = 8.0
    X_val[:, 10] = np.random.uniform(100.0, 1000.0, size=n_val)
    Y_val = np.random.uniform(10.0, 300.0, size=(n_val, 4))

    # Test with curriculum learning, validation data, and early stopping patience
    trainer = PINNTrainer(epochs=30, batch_size=16, patience=10, curriculum=True)
    res = trainer.train(X_train, Y_train, X_val, Y_val)

    assert "best_val_r2" in res
    assert "loss_history" in res
    assert len(res["loss_history"]) > 0
    assert "val_r2" in res["loss_history"][0]
    assert "lr" in res["loss_history"][0]


def test_pinn_evaluator_comprehensive_and_comparison():
    """
    Tests standard metrics, physics consistency scoring, OOD extrapolation evaluation,
    and comparative benchmark of PINN vs standard GA-ANN.
    """
    np.random.seed(42)
    n_samples = 30

    X_test = np.random.uniform(1.0, 10.0, size=(n_samples, 12))
    X_test[:, 3] = np.random.uniform(19.0, 25.0, size=n_samples) # Bench height > 18m
    X_test[:, 6] = np.random.uniform(1.25, 2.00, size=n_samples) # Powder factor > 1.20
    X_test[:, 7] = np.random.uniform(200.0, 800.0, size=n_samples)
    X_test[:, 8] = 8.0
    X_test[:, 10] = np.random.uniform(100.0, 1000.0, size=n_samples)

    Y_test = np.random.uniform(10.0, 300.0, size=(n_samples, 4))

    pinn_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
    evaluator = PINNEvaluator(model=pinn_model)

    in_dist_res = evaluator.evaluate_in_distribution(X_test, Y_test)
    assert "mean_r2" in in_dist_res
    assert "mean_rmse" in in_dist_res
    assert "mean_mae" in in_dist_res

    phys_res = evaluator.evaluate_physics_consistency(X_test)
    assert "physics_consistency_score_pct" in phys_res
    assert "kuzram_d50_mae_mm" in phys_res

    extrap_res = evaluator.evaluate_extrapolation(X_test, Y_test, target_metric="bench_height_gt_18m")
    assert "ood_mean_r2" in extrap_res

    class StandardGAANN:
        def predict(self, x):
            return np.tile([500.0, 50.0, 200.0, 20.0], (len(x), 1))

    comp_res = evaluator.compare_pinn_vs_standard_gaann(
        standard_model=StandardGAANN(),
        ood_x=X_test,
        ood_y=Y_test,
        target_metric="extrapolation_bench_height"
    )

    assert "pinn_metrics" in comp_res
    assert "standard_gaann_metrics" in comp_res
    assert "winner" in comp_res


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
