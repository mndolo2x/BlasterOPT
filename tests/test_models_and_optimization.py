"""
Unit tests for Machine Learning models, prediction wrappers, report generation, and Genetic Algorithm optimizer.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.synthetic_data import generate_synthetic_blast_data
from src.data_ingestion import engineer_features
from src.models import BlastMLPipeline, HAS_TORCH
from src.predict import predict_single_blast, predict_physics_fallback

if HAS_TORCH:
    import torch
    from src.models import GAANNModel
from src.optimize import BlastOptimizer, optimize_blast_design
from src.report import generate_pdf
from src.visualize import (
    plot_kuz_ram_curve,
    plot_ppv_attenuation,
    plot_feature_importance,
    plot_optimization_convergence,
    plot_2d_blast_pattern,
)


@pytest.fixture
def sample_dataset():
    raw_df = generate_synthetic_blast_data(num_samples=100, seed=42)
    return engineer_features(raw_df)


def test_ml_pipeline_train_predict(sample_dataset, tmp_path):
    # Test feature interaction presence
    assert "pf_burden_interaction" in sample_dataset.columns
    assert "spacing_stemming_interaction" in sample_dataset.columns

    # Test Random Forest vs XGBoost Gradient Boosting
    rf_pipeline = BlastMLPipeline(model_type="random_forest", seed=42)
    rf_metrics = rf_pipeline.train_and_evaluate(sample_dataset, cv_folds=3)

    xgb_pipeline = BlastMLPipeline(model_type="xgboost", seed=42)
    xgb_metrics = xgb_pipeline.train_and_evaluate(sample_dataset, cv_folds=3)

    assert "d50_mm" in rf_metrics
    assert "d50_mm" in xgb_metrics
    assert rf_metrics["d50_mm"]["R2_mean"] > -1.0
    assert xgb_metrics["d50_mm"]["R2_mean"] > -1.0

    # Test GridSearchCV hyperparameter tuning & saving to best_fragmentation_model.pkl
    pkl_file = str(tmp_path / "best_fragmentation_model.pkl")
    best_model, best_metrics = xgb_pipeline.tune_and_save_fragmentation_model(
        sample_dataset, save_filepath=pkl_file, cv_folds=3
    )

    assert best_model is not None
    assert "R2" in best_metrics
    assert "RMSE" in best_metrics
    assert "MAE" in best_metrics
    assert os.path.exists(pkl_file)

    # Model persistence
    save_dir = str(tmp_path / "models")
    saved_path = xgb_pipeline.save_models(save_dir)
    assert os.path.exists(saved_path)
    assert os.path.exists(os.path.join(save_dir, "best_fragmentation_model.pkl"))


def test_ga_ann_model():
    """Test GAANNModel instantiation and forward pass tensor output shape."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = GAANNModel(input_size=10)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 3)


def test_optimize_blast_design_torch():
    """Test PyTorch inverse gradient descent optimization function."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = GAANNModel(input_size=10)
    opt_params = optimize_blast_design(model, target_fragmentation=150.0, max_vibration=10.0, max_airblast=120.0)
    assert isinstance(opt_params, torch.Tensor)
    assert opt_params.shape == (10,)


def test_predict_single_blast():
    inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }

    # Fallback
    fb = predict_physics_fallback(inputs)
    assert fb["d50_mm"] > 0
    assert fb["ppv_mms"] > 0
    assert fb["cost_per_tonne_usd"] > 0

    # Single blast predict wrapper
    res = predict_single_blast(inputs, model_pipeline=None)
    assert "d50_mm" in res
    assert "ppv_mms" in res


def test_blast_optimizer_and_report_generation(tmp_path):
    fixed_params = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "monitoring_distance_m": 400.0,
    }

    optimizer = BlastOptimizer(
        fixed_parameters=fixed_params,
        max_ppv_limit_mms=15.0,
        max_flyrock_limit_m=150.0,
        target_d50_range_mm=(100.0, 300.0),
    )

    result = optimizer.optimize(popsize=5, maxiter=5, seed=42)
    assert "optimized_parameters" in result
    assert "predicted_outputs" in result
    assert "top_5_designs" in result
    assert len(result["top_5_designs"]) > 0

    # Test PDF report generation using top 5 designs
    pdf_out = str(tmp_path / "test_report.pdf")
    generated_file = generate_pdf(
        designs=result["top_5_designs"],
        filename=pdf_out,
        constraints_info={"max_ppv": 15.0, "max_flyrock": 150.0, "d50_range": (100.0, 300.0)},
    )

    assert os.path.exists(generated_file)
    assert os.path.getsize(generated_file) > 1000


def test_visualization_functions():
    # Test Kuz-Ram curve generation with default and custom characteristic size (x_c)
    fig1 = plot_kuz_ram_curve(d50_mm=220.0, n_uniformity=1.2)
    assert fig1 is not None
    assert plot_kuz_ram_curve.__doc__ is not None
    assert "Rosin-Rammler" in plot_kuz_ram_curve.__doc__

    fig1_custom = plot_kuz_ram_curve(d50_mm=220.0, n_uniformity=1.5, xc_custom_mm=300.0)
    assert fig1_custom is not None

    fig2 = plot_ppv_attenuation(max_charge_kg=500.0)
    assert fig2 is not None

    df_imp = pd.DataFrame({
        "feature": ["burden_m", "spacing_m"],
        "d50_mm": [0.4, 0.6]
    })
    fig3 = plot_feature_importance(df_imp, target="d50_mm")
    assert fig3 is not None

    fig4 = plot_optimization_convergence([10.0, 8.5, 6.2, 5.1])
    assert fig4 is not None

    fig5 = plot_2d_blast_pattern(num_rows=3, holes_per_row=5)
    assert fig5 is not None
