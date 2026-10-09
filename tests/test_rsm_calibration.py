"""
Tests for RSM Rock Factor Calibration UI results and power-law adjustments.
"""

from src.fragmentation import RockFactorCalibrator


def test_rsm_calibration_harder_rock():
    """Verify rock factor calibration for harder rock (measured > predicted)."""
    meas_d50 = 240.0
    pred_d50 = 200.0
    curr_a = 8.00

    cal_a = RockFactorCalibrator.calibrate_rock_factor(meas_d50, pred_d50, curr_a)
    ratio = meas_d50 / pred_d50
    adjustment_pct = ((ratio ** 0.8) - 1.0) * 100.0

    assert cal_a == 9.26
    assert round(ratio, 3) == 1.200
    assert round(adjustment_pct, 1) == 15.7
    assert cal_a > curr_a


def test_rsm_calibration_softer_rock():
    """Verify rock factor calibration for softer rock (measured < predicted)."""
    meas_d50 = 160.0
    pred_d50 = 200.0
    curr_a = 8.00

    cal_a = RockFactorCalibrator.calibrate_rock_factor(meas_d50, pred_d50, curr_a)
    ratio = meas_d50 / pred_d50
    adjustment_pct = ((ratio ** 0.8) - 1.0) * 100.0

    assert cal_a < curr_a
    assert ratio < 1.0
    assert adjustment_pct < 0.0
