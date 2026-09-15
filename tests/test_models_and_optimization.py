"""
Unit tests for Machine Learning models, prediction wrappers, and Genetic Algorithm optimizer.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.synthetic_data import generate_synthetic_blast_data
from src.data_ingestion import engineer_features
from src.models import BlastMLPipeline
from src.predict import predict_single_blast, predict_physics_fallback
from src.optimize import BlastOptimizer
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
    pipeline = BlastMLPipeline(model_type="random_forest", seed=42)
    metrics = pipeline.train_and_evaluate(sample_dataset, cv_folds=3)

    assert "d50_mm" in metrics
    assert "ppv_mms" in metrics
    assert metrics["d50_mm"]["R2_mean"] > -1.0

    # Predictions
    preds = pipeline.predict(sample_dataset.head(5))
    assert len(preds) == 5
    assert "pred_d50_mm" in preds.columns
    assert "pred_ppv_mms" in preds.columns

    # Model persistence
    save_dir = str(tmp_path / "models")
    saved_path = pipeline.save_models(save_dir)
    assert os.path.exists(saved_path)

    loaded_pipeline = BlastMLPipeline.load_models(saved_path)
    assert loaded_pipeline.model_type == "random_forest"


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


def test_blast_optimizer():
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
    assert result["optimized_parameters"]["burden_m"] >= 2.5
    assert result["optimized_parameters"]["spacing_m"] >= 3.0


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
