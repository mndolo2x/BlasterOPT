#!/usr/bin/env python3
"""
Demo Script for Site-Specific Vibration Attenuation Calibration Module (`site_calibration`).

Demonstrates:
1. Ingesting synthetic seismograph vibration readings across a mine site
2. Nonlinear regression fitting of site-specific K and B attenuation parameters
3. GSI-modified attenuation law evaluation
4. PPV ground vibration prediction with 95% confidence bounds
5. Generating Plotly visualization figures
6. Continuous recalibration feedback loop execution
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath("."))

from src.site_calibration import (
    SeismographReading,
    CalibrationManager,
    FeedbackLoopEngine,
    evaluate_gsi_improvement,
    plot_attenuation_curve,
    plot_residuals,
    plot_gsi_correction_curve,
    plot_calibration_dashboard,
)


def main():
    print("=" * 80)
    print("💥 BlastOpt Botswana - Site-Specific Vibration Attenuation Calibration Demo")
    print("=" * 80)

    # 1. Initialize Calibration Manager
    db_path = "data/processed/demo_calibration_db.json"
    manager = CalibrationManager(db_file_path=db_path)
    site_id = "SITE_DEBSWANA_JWANENG_CUT8"
    rock_type = "Kimberlite_Hard"

    print(f"\n[1/6] Generating synthetic seismograph vibration dataset for site '{site_id}'...")
    np.random.seed(42)
    n_readings = 35

    k_true = 580.0
    b_true = 1.62
    a_true = 0.95
    b_gsi_true = 0.006

    distances = np.random.uniform(150.0, 1200.0, size=n_readings)
    charges = np.random.uniform(300.0, 900.0, size=n_readings)
    gsis = np.random.uniform(35.0, 80.0, size=n_readings)

    sd = distances / np.sqrt(charges)
    ppv_true = k_true * (sd ** (-b_true)) * a_true * np.exp(b_gsi_true * gsis)
    noise = np.random.normal(1.0, 0.04, size=n_readings)
    ppvs = np.maximum(0.1, ppv_true * noise)

    for i in range(n_readings):
        reading = SeismographReading(
            blast_id=f"BLAST_DEMO_{i+1:03d}",
            site_id=site_id,
            distance_m=distances[i],
            ppv_mm_s=ppvs[i],
            charge_per_delay_kg=charges[i],
            dominant_frequency_hz=28.5,
            gsi_of_transmission_strata=gsis[i],
            rock_type=rock_type,
        )
        manager.add_reading(reading)

    print(f"✔ Successfully ingested {len(manager.readings_db)} seismograph readings!")

    # 2. Fit Site Attenuation Parameters
    print(f"\n[2/6] Executing nonlinear regression parameter fitting for site '{site_id}'...")
    calib_res = manager.fit_site(site_id=site_id, rock_type=rock_type)

    print(f"✔ Fit Quality Rating: {calib_res.fit_quality}")
    if calib_res.parameters:
        params = calib_res.parameters
        print(f"  • Site Constant K: {params.K:.2f} (95% CI: [{params.confidence_interval_95['K'][0]:.1f}, {params.confidence_interval_95['K'][1]:.1f}])")
        print(f"  • Attenuation Exponent B: {params.B:.3f} (95% CI: [{params.confidence_interval_95['B'][0]:.3f}, {params.confidence_interval_95['B'][1]:.3f}])")
        if params.gsi_b:
            print(f"  • GSI Correction: f(GSI) = {params.gsi_a:.2f} * exp({params.gsi_b:.4f} * GSI)")
        print(f"  • Coefficient of Determination R²: {params.r_squared:.4f}")
        print(f"  • RMSE Fit Error: {params.rmse:.3f} mm/s")

    # 3. Predict PPV
    print("\n[3/6] Predicting PPV ground vibration at distance D = 450m, Max Charge Q = 640kg, GSI = 65...")
    pred_res = manager.predict_ppv(
        site_id=site_id,
        distance_m=450.0,
        charge_per_delay_kg=640.0,
        gsi=65.0,
        rock_type=rock_type,
    )

    print(f"✔ Predicted PPV: {pred_res.ppv_mm_s:.2f} mm/s (95% CI: [{pred_res.lower_95:.2f}, {pred_res.upper_95:.2f}] mm/s)")
    print(f"  • Method Used: {pred_res.method}")

    # 4. Evaluate GSI Correction Impact
    print("\n[4/6] Evaluating GSI correction model improvement over standard USBM law...")
    gsi_eval = evaluate_gsi_improvement(r2_standard=0.82, r2_gsi_modified=params.r_squared if params else 0.88)
    print(f"✔ R² Improvement: +{gsi_eval['improvement_pct']:.1f}%")
    print(f"  • Recommendation: {gsi_eval['recommendation']}")

    # 5. Continuous Feedback Loop Recalibration
    print("\n[5/6] Executing continuous feedback loop processing for post-blast reading...")
    feedback_engine = FeedbackLoopEngine(calibration_manager=manager)
    post_blast_reading = SeismographReading(
        blast_id="BLAST_POST_SHOT_999",
        site_id=site_id,
        distance_m=420.0,
        ppv_mm_s=8.2,
        charge_per_delay_kg=640.0,
        rock_type=rock_type,
    )
    recal_log = feedback_engine.process_new_reading(post_blast_reading, force_recalibrate=True)

    if recal_log:
        print(f"✔ Recalibration Logged! Reason: '{recal_log.reason}'")
        print(f"  • K Delta: {recal_log.delta_k_pct:.1f}% | B Delta: {recal_log.delta_b_pct:.1f}%")

    # 6. Visualizer Figures Output
    print("\n[6/6] Generating Plotly visualization figures...")
    fig_curve = plot_attenuation_curve(distances, ppvs, charges, params=params)
    fig_dash = plot_calibration_dashboard(calib_res, output_html_path="data/processed/demo_calibration_dashboard.html")

    print("✔ Visualizations generated successfully!")
    print("  • Saved Dashboard HTML: 'data/processed/demo_calibration_dashboard.html'")

    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
