"""
Unit tests for FragmentationCalibrator in src/fragmentation/calibration.py.
"""

import pytest
from src.fragmentation import FragmentationCalibrator


def test_fragmentation_calibrator():
    calibrator = FragmentationCalibrator()

    bp = {
        "powder_factor_kg_m3": 0.65,
        "charge_kg": 150.0,
        "rock_factor_a": 8.0,
        "blastability_index": 50.0,
    }

    for _ in range(5):
        calibrator.add_blast_record(bp, measured_d80_cm=28.0)

    cal_res = calibrator.calibrate_rock_factor()
    assert "beta_1" in cal_res
    assert "r_squared" in cal_res

    pred = calibrator.predict_calibrated_d80(bp)
    assert "d80_cm" in pred
    assert "confidence_interval" in pred

    quality = calibrator.get_calibration_quality()
    assert "n_samples" in quality
    assert "is_reliable" in quality
    assert quality["n_samples"] == 5
    assert quality["is_reliable"] is False  # < 30 samples
