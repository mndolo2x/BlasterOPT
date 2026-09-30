"""
Unit tests for fragmentation module verification per user prompt specifications.
"""

import pytest
import numpy as np
import pandas as pd
from src.fragmentation import (
    SwebrecModel,
    KCOModel,
    predict_kuz_ram,
    compare_distributions,
    ImageAnalysisImporter,
    FragmentationCalibrator,
)


def test_swebrec_formula():
    """Verify Swebrec P(x) at x=x_50 equals 50%."""
    model = SwebrecModel(x_max=100.0, x_50=30.0, b=2.0)
    assert abs(model.percent_passing(30.0) - 50.0) < 0.01


def test_swebrec_at_x_max():
    """Verify Swebrec P(x_max) approaches 100%."""
    model = SwebrecModel(x_max=100.0, x_50=30.0, b=2.0)
    assert model.percent_passing(100.0) > 99.0


def test_kco_x50_matches_kuz_ram():
    """KCO x_50 must match Kuz-Ram x_50 (same Kuznetsov equation)."""
    bp = {
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "powder_factor_kg_m3": 0.65,
        "explosive_rws": 100.0,
        "rock_factor_a": 8.0,
    }
    kco = KCOModel()
    kco_res = kco.predict(bp)

    vol = 4.0 * 5.0 * 12.0
    kr_res = predict_kuz_ram(0.65, vol * 0.65, rock_factor_a=8.0, explosive_relative_weight_strength=100.0)

    assert abs(kco_res["x_50_cm"] - kr_res["d50_cm"]) < 0.1


def test_distribution_comparison_returns_three_curves():
    """compare_distributions must return three DataFrames."""
    result = compare_distributions(x_50=25.0, n_uniformity=1.2, b_swebrec=0.8, x_max=100.0)
    assert "rosin_rammler" in result
    assert "swebrec" in result
    assert "lognormal" in result


def test_image_analysis_import():
    """WipFrag CSV import must return a valid DataFrame."""
    importer = ImageAnalysisImporter()
    df = importer.import_wipfrag_csv("tests/fixtures/wipfrag_sample.csv")
    assert "size_mm" in df.columns
    assert "percent_passing" in df.columns


def test_calibration_improves_accuracy():
    """RSM calibration must reduce RMSE vs. uncalibrated Kuz-Ram."""
    calibrator = FragmentationCalibrator()
    for i in range(50):
        calibrator.add_blast_record(
            {
                "powder_factor_kg_m3": 0.6 + i * 0.01,
                "charge_kg": 300.0,
                "rock_factor_a": 8.0,
                "blastability_index": 60.0,
            },
            measured_d80_cm=25.0 + i * 0.1,
        )
    result = calibrator.calibrate_rock_factor()
    assert result["improvement_pct"] > 0
    assert result["n_samples"] == 50
