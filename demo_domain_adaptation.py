#!/usr/bin/env python3
"""
Demo Script for Geology Domain Adaptation Module (`src/domain_adaptation`).

Demonstrates:
1. Generating Kimberlite (Source) and Granite (Target) domain datasets.
2. Evaluating Kimberlite zero-shot model accuracy degradation on harder Granite geology.
3. Layer-freezing transfer learning fine-tuning on Granite target data.
4. Adversarial DANN feature alignment using Gradient Reversal Layer (GRL).
5. Generating Plotly PCA domain feature alignment and $R^2$ accuracy comparison plots.
"""

import sys
import os
import numpy as np

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath("."))

from src.physics_informed import PhysicsInformedGAANN
from src.domain_adaptation import (
    DomainAdaptationConfig,
    DomainDataManager,
    DomainAdaptationManager,
    plot_domain_feature_distribution,
    plot_adaptation_r2_comparison,
)


def main():
    print("=" * 80)
    print("💎 BlastOpt Botswana - Geology Domain Adaptation (Kimberlite -> Granite) Demo")
    print("=" * 80)

    # 1. Generate Kimberlite (Source) and Granite (Target) datasets
    print("\n[1/5] Loading Kimberlite (Source) and Granite (Target) domain datasets...")
    data_mgr = DomainDataManager()
    X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=200, n_target=40, seed=42)

    print(f"✔ Source (Kimberlite) Samples: {X_src.shape[0]} | Target (Granite) Samples: {X_tgt.shape[0]}")
    print(f"✔ Kimberlite Rock Factor A: 8.0 | Granite Rock Factor A: 11.0 (Harder Rock)")

    # 2. Base Kimberlite model
    print("\n[2/5] Instantiating base Kimberlite blast prediction neural network...")
    base_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)

    # 3. Domain Adaptation Orchestration
    print("\n[3/5] Executing Domain Adaptation Manager (Layer Freezing Fine-Tuning + DANN)...")
    config = DomainAdaptationConfig(
        source_geology="Kimberlite",
        target_geology="Granite",
        freeze_early_layers=True,
        fine_tune_epochs=50,
        fine_tune_lr=0.001,
        target_r2_threshold=0.70
    )

    adaptation_mgr = DomainAdaptationManager(config=config)
    adapted_model, report = adaptation_mgr.adapt_domain(base_model, X_src, Y_src, X_tgt, Y_tgt)

    # 4. Print Adaptation Results Report
    print("\n[4/5] Domain Adaptation Executive Summary:")
    print("-" * 60)
    print(f"• Pre-Adaptation Zero-Shot Target R²:  {report.metrics.pre_adaptation_r2:.4f}")
    print(f"• Post-Adaptation Target R²:           {report.metrics.post_adaptation_r2:.4f}")
    print(f"• Target R² Improvement:               +{report.metrics.r2_improvement:.4f}")
    print(f"• Target Domain RMSE:                  {report.metrics.target_rmse:.2f}")
    print(f"• Adaptation Strategy Selected:        {report.metrics.method_used}")
    print(f"• Adaptation Status:                   {report.status}")
    print("-" * 60)

    print("\nEngineering Recommendations for Granite Geology:")
    for idx, rec in enumerate(report.recommendations, 1):
        print(f"  {idx}. {rec}")

    # 5. Generate Visualizations
    print("\n[5/5] Generating Plotly domain feature distribution & accuracy plots...")
    fig_pca = plot_domain_feature_distribution(X_src, X_tgt)
    fig_r2 = plot_adaptation_r2_comparison(report.metrics.pre_adaptation_r2, report.metrics.post_adaptation_r2)

    print("✔ Visualizations created successfully!")

    print("\n" + "=" * 80)
    print("✅ Kimberlite -> Granite Geology Domain Adaptation Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
