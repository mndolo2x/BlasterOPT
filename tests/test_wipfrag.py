"""
Tests for WipFrag optical fragmentation data input and comparison against Kuz-Ram prediction.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def test_wipfrag_manual_entry_dataframe_construction():
    """Verify building 5-point distribution from manual percentile inputs."""
    d20 = 45.0
    d50 = 210.0
    d80 = 420.0
    fines = 8.5
    boulders = 4.2

    measured_data = {
        "size_mm": [10, d20, d50, d80, 800],
        "percent_passing": [fines, 20, 50, 80, 100 - boulders],
    }
    df = pd.DataFrame(measured_data)

    assert len(df) == 5
    assert list(df["size_mm"]) == [10, 45.0, 210.0, 420.0, 800]
    assert list(df["percent_passing"]) == [8.5, 20, 50, 80, 95.8]


def test_wipfrag_error_metric_calculation():
    """Verify RMSE, MAE, and Bias calculations between measured and predicted curves."""
    x50_pred_cm = 21.0
    n_pred = 1.25

    # Measured data points
    measured_data = {
        "size_mm": [10, 45.0, 210.0, 420.0, 800],
        "percent_passing": [8.5, 20.0, 50.0, 80.0, 95.8],
    }
    measured_df = pd.DataFrame(measured_data)

    x_pred_cm = np.linspace(0.5, 100, 500)
    predicted_passing = 100 * (1 - np.exp(-0.693 * (x_pred_cm / x50_pred_cm) ** n_pred))

    x_meas_cm = measured_df["size_mm"].values / 10.0
    y_meas = measured_df["percent_passing"].values

    pred_interp = interp1d(x_pred_cm, predicted_passing, bounds_error=False, fill_value="extrapolate")
    predicted_at_measured = pred_interp(x_meas_cm)
    residuals = y_meas - predicted_at_measured

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))

    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert isinstance(bias, float)
    assert rmse >= 0.0
    assert mae >= 0.0


def test_wipfrag_d50_shift_impacts_residuals():
    """Verify that changing measured d50 alters error metrics as expected."""
    x50_pred_cm = 21.0
    n_pred = 1.25
    x_pred_cm = np.linspace(0.5, 100, 500)
    predicted_passing = 100 * (1 - np.exp(-0.693 * (x_pred_cm / x50_pred_cm) ** n_pred))
    pred_interp = interp1d(x_pred_cm, predicted_passing, bounds_error=False, fill_value="extrapolate")

    # Entry 1: d50 = 210mm (matching predicted d50)
    df1 = pd.DataFrame({
        "size_mm": [10, 45.0, 210.0, 420.0, 800],
        "percent_passing": [8.5, 20.0, 50.0, 80.0, 95.8],
    })
    x_meas_cm1 = df1["size_mm"].values / 10.0
    bias1 = float(np.mean(df1["percent_passing"].values - pred_interp(x_meas_cm1)))

    # Entry 2: d50 = 350mm (coarser measured distribution)
    df2 = pd.DataFrame({
        "size_mm": [10, 45.0, 350.0, 420.0, 800],
        "percent_passing": [8.5, 20.0, 50.0, 80.0, 95.8],
    })
    x_meas_cm2 = df2["size_mm"].values / 10.0
    bias2 = float(np.mean(df2["percent_passing"].values - pred_interp(x_meas_cm2)))

    assert bias1 != bias2
