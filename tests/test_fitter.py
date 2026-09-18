"""
Unit tests for Attenuation Fitter (src/site_calibration/attenuation_fitter.py).
"""

import pytest
import numpy as np
from src.site_calibration.attenuation_fitter import AttenuationFitter


def test_fitter_synthetic_data_recovery():
    """
    Generates synthetic seismograph data with known parameters:
    K = 500.0, B = 1.6
    PPV = 500 * (D / sqrt(Q))^(-1.6)
    Verifies that AttenuationFitter recovers K and B within 10% tolerance.
    """
    np.random.seed(42)

    k_true = 500.0
    b_true = 1.6

    n_samples = 40
    distances = np.random.uniform(100.0, 1500.0, size=n_samples)
    charges = np.random.uniform(200.0, 1000.0, size=n_samples)

    sd = distances / np.sqrt(charges)
    ppv_true = k_true * (sd ** (-b_true))

    # Add Gaussian noise (log-normal multiplicative noise)
    noise_factor = np.random.normal(1.0, 0.05, size=n_samples)
    ppv_noisy = np.maximum(0.1, ppv_true * noise_factor)

    fitter = AttenuationFitter(min_samples=10, min_r2=0.70)
    res = fitter.fit_usbm(distances, ppv_noisy, charges, rock_type="Kimberlite")

    assert res.fit_quality in ["EXCELLENT", "ACCEPTABLE"]
    assert res.parameters is not None

    fitted_k = res.parameters.K
    fitted_b = res.parameters.B

    # Verify recovery within 10%
    assert abs(fitted_k - k_true) / k_true < 0.10
    assert abs(fitted_b - b_true) / b_true < 0.10
    assert res.parameters.r_squared >= 0.85


def test_fitter_rejection_low_r2():
    """
    Generates pure random noise where R2 < 0.70 and verifies rejection.
    """
    np.random.seed(42)
    n_samples = 20
    distances = np.random.uniform(100.0, 1500.0, size=n_samples)
    charges = np.random.uniform(200.0, 1000.0, size=n_samples)
    ppv_pure_noise = np.random.uniform(1.0, 50.0, size=n_samples)

    fitter = AttenuationFitter(min_samples=10, min_r2=0.70)
    res = fitter.fit_usbm(distances, ppv_pure_noise, charges)

    assert res.fit_quality == "REJECTED"
    assert len(res.warnings) >= 1
    assert any("below minimum threshold" in w for w in res.warnings)


def test_fitter_bootstrap_confidence_interval():
    """
    Verifies that bootstrap 95% confidence intervals are generated and reasonable.
    """
    np.random.seed(42)
    n_samples = 30
    distances = np.random.uniform(100.0, 1000.0, size=n_samples)
    charges = np.random.uniform(300.0, 800.0, size=n_samples)
    sd = distances / np.sqrt(charges)
    ppv = 600.0 * (sd ** (-1.5)) * np.random.normal(1.0, 0.03, size=n_samples)

    fitter = AttenuationFitter(min_samples=10, bootstrap_iters=200)
    res = fitter.fit_usbm(distances, ppv, charges)

    assert res.parameters is not None
    ci = res.parameters.confidence_interval_95
    assert "K" in ci
    assert "B" in ci
    assert ci["K"][0] < res.parameters.K < ci["K"][1]
    assert ci["B"][0] < res.parameters.B < ci["B"][1]
