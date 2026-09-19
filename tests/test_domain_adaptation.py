"""
Unit tests for Geology Domain Adaptation Package (`src/domain_adaptation`).
"""

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.domain_adaptation import (
    DomainAdaptationConfig,
    DomainDataManager,
    TransferFineTuner,
    JDAAligner,
    compute_rbf_mmd,
    DomainAdversarialGAANN,
    DANNTrainer,
    CrossDomainEvaluator,
    DomainAdaptationManager,
    plot_domain_feature_distribution,
    plot_adaptation_r2_comparison,
)
from src.physics_informed import PhysicsInformedGAANN


def test_domain_data_manager():
    """
    Test dataset generation and feature alignment for Kimberlite and Granite geologies.
    """
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=50, n_target=20, seed=42)

    assert X_src.shape == (50, 12)
    assert Y_src.shape == (50, 4)
    assert X_tgt.shape == (20, 12)
    assert Y_tgt.shape == (20, 4)

    # Verify Granite rock factor A = 11.0 and Kimberlite A = 8.0
    assert np.allclose(X_src[:, 8], 8.0)
    assert np.allclose(X_tgt[:, 8], 11.0)


def test_jda_aligner():
    """
    Test Joint Domain Adaptation (JDA) MMD distance and feature projection.
    """
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=40, n_target=20, seed=42)

    orig_mmd = compute_rbf_mmd(X_src, X_tgt)
    assert orig_mmd >= 0.0

    jda = JDAAligner(n_components=6)
    Z_src, Z_tgt, info = jda.fit_transform(X_src, X_tgt)

    assert Z_src.shape == (40, 6)
    assert Z_tgt.shape == (20, 6)
    assert "aligned_mmd" in info


def test_transfer_fine_tuner():
    """
    Test transfer fine-tuning layer freezing and target adaptation.
    """
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=40, n_target=20, seed=42)

    base_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
    fine_tuner = TransferFineTuner(lr=0.001, epochs=10, freeze_early_layers=True)

    ft_model, info = fine_tuner.fine_tune(base_model, X_tgt, Y_tgt)

    assert ft_model is not None
    assert info["fine_tune_epochs"] == 10
    assert "final_loss" in info


def test_dann_adversarial_trainer():
    """
    Test DANN feature alignment and Gradient Reversal Layer.
    """
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed.")

    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=40, n_target=20, seed=42)

    trainer = DANNTrainer(epochs=10, lr=0.001, alpha_grl=1.0)
    dann_model, info = trainer.fit(X_src, Y_src, X_tgt, Y_tgt)

    assert dann_model is not None
    assert info["epochs"] == 10
    assert len(info["loss_history"]) == 10

    # Test forward pass returns task predictions and domain logits
    x_t = torch.tensor(X_tgt[:5], dtype=torch.float32)
    task_preds, domain_logits = dann_model(x_t)
    assert task_preds.shape == (5, 4)
    assert domain_logits.shape == (5, 1)


def test_cross_domain_evaluator():
    """
    Test CrossDomainEvaluator MMD shift calculation and accuracy metrics.
    """
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=40, n_target=20, seed=42)

    evaluator = CrossDomainEvaluator(source_name="Kimberlite", target_name="Granite")
    shift_metrics = evaluator.evaluate_domain_shift(X_src, X_tgt)

    assert shift_metrics.mmd_distance >= 0.0
    assert shift_metrics.domain_shift_level in ["LOW", "MODERATE", "HIGH"]

    base_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
    r2, rmse, mae = evaluator.evaluate_accuracy(base_model, X_tgt, Y_tgt)
    assert rmse >= 0.0
    assert mae >= 0.0


def test_domain_adaptation_manager_orchestration():
    """
    Test DomainAdaptationManager full workflow from Kimberlite to Granite.
    """
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=50, n_target=25, seed=42)

    base_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)
    config = DomainAdaptationConfig(
        source_geology="Kimberlite",
        target_geology="Granite",
        freeze_early_layers=True,
        fine_tune_epochs=15,
        target_r2_threshold=0.99  # Force trigger DANN
    )

    manager = DomainAdaptationManager(config=config)
    adapted_model, report = manager.adapt_domain(base_model, X_src, Y_src, X_tgt, Y_tgt)

    assert adapted_model is not None
    assert report.source_domain == "Kimberlite"
    assert report.target_domain == "Granite"
    assert report.metrics.post_adaptation_r2 is not None
    assert len(report.recommendations) > 0


def test_domain_adaptation_visualizers():
    """
    Test Plotly visualization figures in src/domain_adaptation/visualizer.py.
    """
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=30, n_target=15, seed=42)

    fig_pca = plot_domain_feature_distribution(X_src, X_tgt)
    assert fig_pca is not None

    fig_r2 = plot_adaptation_r2_comparison(pre_adaptation_r2=0.45, post_adaptation_r2=0.82)
    assert fig_r2 is not None
