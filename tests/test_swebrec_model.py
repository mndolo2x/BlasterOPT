"""
Unit tests for SwebrecModel in src/fragmentation/swebrec.py.
"""

import pytest
import numpy as np
from src.fragmentation import SwebrecModel


def test_swebrec_model():
    model = SwebrecModel(x_max=1000.0, x_50=250.0, b=1.2)
    assert abs(model.percent_passing(250.0) - 50.0) < 0.1
    assert model.percent_passing(1000.0) == 100.0
    assert model.percent_passing(0.0) == 0.0

    df = model.compute_curve(x_min=1.0, x_max=1000.0, n_points=50)
    assert len(df) == 50
    assert "size_mm" in df.columns
    assert "percent_passing" in df.columns

    b_est = model.calculate_b_from_uniformity(n_uniformity=1.25)
    assert b_est == 0.625

    x_max_est = model.calculate_x_max_from_burden(burden_m=4.0, spacing_m=5.0)
    assert x_max_est == 4000.0


def test_swebrec_fit():
    sizes = np.array([10.0, 50.0, 100.0, 250.0, 500.0, 800.0])
    passings = np.array([10.0, 25.0, 35.0, 50.0, 75.0, 90.0])

    model = SwebrecModel(x_max=1000.0, x_50=250.0, b=1.0)
    fit_res = model.fit_from_data(sizes, passings)

    assert "x_max" in fit_res
    assert "x_50" in fit_res
    assert "b" in fit_res
    assert "r_squared" in fit_res
