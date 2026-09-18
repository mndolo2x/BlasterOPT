"""
Unit tests for GSI Correction (src/site_calibration/gsi_correction.py).
"""

import pytest
import numpy as np
from src.site_calibration.models import AttenuationParameters
from src.site_calibration.gsi_correction import (
    calculate_gsi_factor,
    predict_gsi_modified_ppv,
    evaluate_gsi_improvement,
)
from src.site_calibration.attenuation_fitter import AttenuationFitter


def test_gsi_factor_calculation():
    """Test calculate_gsi_factor exponential multiplier f(GSI) = a * exp(b * GSI)."""
    val = calculate_gsi_factor(gsi=60.0, a=1.0, b=0.01)
    expected = 1.0 * np.exp(0.01 * 60.0)
    assert abs(val - expected) < 1e-4


def test_gsi_prediction_fallback():
    """
    Test predict_gsi_modified_ppv falls back to standard USBM when GSI is None or gsi_b is 0.0.
    """
    params = AttenuationParameters(
        K=500.0,
        B=1.6,
        gsi_a=1.0,
        gsi_b=0.0,
        r_squared=0.90,
        rmse=1.2,
        sample_count=20,
    )

    pred_fallback = predict_gsi_modified_ppv(
        distance_m=450.0,
        charge_per_delay_kg=640.0,
        params=params,
        gsi=None,
    )

    assert pred_fallback.method == "USBM_SITE_CALIBRATED"
    sd = 450.0 / (640.0 ** 0.5)
    expected_ppv = 500.0 * (sd ** (-1.6))
    assert abs(pred_fallback.ppv_mm_s - round(expected_ppv, 2)) < 0.1


def test_gsi_fit_and_recovery():
    """
    Generates synthetic data with known GSI correction:
    PPV = 500 * (D/sqrt(Q))^(-1.6) * 1.0 * exp(0.008 * GSI)
    Fits and verifies parameter recovery.
    """
    np.random.seed(42)
    n_samples = 30
    distances = np.random.uniform(100.0, 1000.0, size=n_samples)
    charges = np.random.uniform(200.0, 800.0, size=n_samples)
    gsis = np.random.uniform(30.0, 80.0, size=n_samples)

    sd = distances / np.sqrt(charges)
    ppv_true = 500.0 * (sd ** (-1.6)) * 1.0 * np.exp(0.008 * gsis)

    fitter = AttenuationFitter()
    res = fitter.fit_gsi_modified(distances, ppv_true, charges, gsis)

    assert res.parameters is not None
    assert res.parameters.r_squared >= 0.85
    assert abs(res.parameters.gsi_b - 0.008) < 0.01


def test_evaluate_gsi_improvement():
    """Test evaluate_gsi_improvement computing R2 improvement percentage."""
    eval_res = evaluate_gsi_improvement(r2_standard=0.80, r2_gsi_modified=0.88)
    assert eval_res["improvement_pct"] == 10.0
    assert eval_res["is_significant"] is True
    assert "Adopt GSI-modified" in eval_res["recommendation"]
