#!/usr/bin/env python3
"""
Demo Script for Ensemble Uncertainty Quantification & OOD Detection (`ensemble_uq`).

Demonstrates:
1. Training N=7 GA-ANN bagging ensemble members across bootstrap sub-samples.
2. Predicting blast outcomes on in-distribution inputs (normal powder factor & bench height).
3. Predicting blast outcomes on Out-Of-Distribution (OOD) inputs (powder factor > 1.20 kg/m³, bench height > 18m).
4. Decomposing uncertainty into aleatoric (data noise) vs. epistemic (model knowledge gap).
5. Generating Plotly visualization figures for uncertainty and OOD Mahalanobis distributions.
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath("."))

from src.ensemble_uq import (
    TrainingConfig,
    UncertaintyAwarePredictor,
    plot_prediction_intervals,
    plot_uncertainty_decomposition,
    plot_ood_distribution,
    plot_confidence_calibration,
)


def main():
    print("=" * 80)
    print("🛡️ BlastOpt Botswana - GA-ANN Ensemble Uncertainty Quantification (UQ) Demo")
    print("=" * 80)

    # 1. Generate synthetic training dataset
    print("\n[1/5] Generating training dataset for GA-ANN ensemble...")
    np.random.seed(42)
    n_samples = 200
    n_features = 6
    feature_names = ["burden_m", "spacing_m", "bench_height_m", "powder_factor_kg_m3", "max_charge_per_delay_kg", "monitoring_distance_m"]

    X = np.column_stack([
        np.random.uniform(3.0, 9.0, size=n_samples),      # burden_m
        np.random.uniform(4.0, 11.0, size=n_samples),     # spacing_m
        np.random.uniform(8.0, 16.0, size=n_samples),     # bench_height_m
        np.random.uniform(0.30, 0.95, size=n_samples),    # powder_factor_kg_m3
        np.random.uniform(200.0, 900.0, size=n_samples),  # max_charge_per_delay_kg
        np.random.uniform(100.0, 1000.0, size=n_samples) # monitoring_distance_m
    ])

    d50 = 250.0 - 120.0 * X[:, 3] + 8.0 * X[:, 0] + np.random.normal(0, 5.0, size=n_samples)
    ppv = 1140.0 * ((X[:, 5] / np.sqrt(X[:, 4])) ** -1.6) + np.random.normal(0, 0.3, size=n_samples)
    flyrock = 80.0 + 45.0 * X[:, 3] + np.random.normal(0, 3.0, size=n_samples)
    cost = 3.50 + 2.0 * X[:, 3] + 0.15 * X[:, 0] + np.random.normal(0, 0.1, size=n_samples)
    Y = np.column_stack([d50, ppv, flyrock, cost])

    # 2. Train N=7 GA-ANN Ensemble
    print("\n[2/5] Training N=7 GA-ANN bagging ensemble members...")
    config = TrainingConfig(n_models=7, seeds=[42, 101, 202, 303, 404, 505, 606])
    predictor = UncertaintyAwarePredictor(config=config, model_dir="data/processed/ensemble_models/")
    predictor.fit_and_train(X, Y, feature_names=feature_names)
    print("✔ Ensemble training complete!")

    # 3. Predict In-Distribution Sample
    print("\n[3/5] Evaluating In-Distribution Sample (PF = 0.65 kg/m³, Height = 12m)...")
    normal_input = {
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "bench_height_m": 12.0,
        "powder_factor_kg_m3": 0.65,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }
    pred_normal = predictor.predict(normal_input)

    print(f"✔ Prediction Means: {pred_normal.mean}")
    print(f"✔ 95% Confidence Bounds (d50_mm): [{pred_normal.lower_95['d50_mm']:.1f}, {pred_normal.upper_95['d50_mm']:.1f}] mm")
    print(f"✔ Is OOD: {pred_normal.is_ood} | Severity: {pred_normal.ood_report.severity if pred_normal.ood_report else 'NORMAL'}")
    print(f"✔ Natural Language Explanation:\n  {predictor.explain_uncertainty(pred_normal)}")

    # 4. Predict Out-Of-Distribution (OOD) Sample
    print("\n[4/5] Evaluating Out-Of-Distribution (OOD) Sample (PF = 1.45 kg/m³ > 1.20, Height = 22m > 18m)...")
    ood_input = {
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "bench_height_m": 22.0,          # Exceeds max 18m
        "powder_factor_kg_m3": 1.45,       # Exceeds max 1.20
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }
    pred_ood = predictor.predict(ood_input)

    print(f"✔ Is OOD: {pred_ood.is_ood} | Severity: {pred_ood.ood_report.severity if pred_ood.ood_report else 'CRITICAL'}")
    print(f"✔ Exceeded Thresholds: {pred_ood.ood_report.threshold_exceeded if pred_ood.ood_report else []}")
    print(f"✔ High Uncertainty Flagged: {predictor.flag_high_uncertainty(pred_ood)}")
    print(f"✔ Natural Language OOD Explanation:\n  {predictor.explain_uncertainty(pred_ood)}")

    # 5. Generate Plotly Figures
    print("\n[5/5] Generating Plotly visualization figures...")
    fig_decomp = plot_uncertainty_decomposition(pred_ood)
    fig_calib = plot_confidence_calibration(actual_coverage=0.95, target_coverage=0.95)

    print("✔ Visualizations generated successfully!")

    print("\n" + "=" * 80)
    print("✅ GA-ANN Ensemble UQ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
