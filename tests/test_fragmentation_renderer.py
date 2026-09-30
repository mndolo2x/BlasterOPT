"""
Unit tests for renderer functions in src/fragmentation/renderer.py.
"""

import pytest
import numpy as np
import pandas as pd
from src.fragmentation import (
    plot_fragmentation_curve,
    plot_model_comparison,
    plot_calibration_scatter,
)


def test_plot_fragmentation_curve():
    df_pred = pd.DataFrame({"size_mm": [10, 50, 100, 200], "percent_passing": [10, 30, 60, 90]})
    df_meas = pd.DataFrame({"size_mm": [10, 50, 100, 200], "percent_passing": [12, 28, 62, 88]})

    fig = plot_fragmentation_curve(df_pred, model_name="KCO", measured_df=df_meas)
    assert fig is not None


def test_plot_calibration_scatter():
    meas = np.array([20.0, 25.0, 30.0, 35.0])
    pred_base = np.array([22.0, 28.0, 34.0, 39.0])
    pred_cal = np.array([20.5, 25.2, 29.8, 35.1])

    fig = plot_calibration_scatter(meas, pred_base, pred_cal)
    assert fig is not None
