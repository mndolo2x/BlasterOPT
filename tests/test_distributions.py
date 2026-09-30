"""
Unit tests for distributions comparison in src/fragmentation/distributions.py.
"""

import pytest
import pandas as pd
from src.fragmentation import (
    rosin_rammler_curve,
    lognormal_curve,
    compare_distributions,
    plot_distribution_comparison,
    calculate_goodness_of_fit,
)


def test_rosin_rammler_curve():
    df = rosin_rammler_curve(x_50=250.0, n=1.25)
    assert isinstance(df, pd.DataFrame)
    assert "size_mm" in df.columns
    assert "percent_passing" in df.columns


def test_lognormal_curve():
    df = lognormal_curve(x_50=250.0, sigma=0.6)
    assert isinstance(df, pd.DataFrame)
    assert "size_mm" in df.columns


def test_compare_distributions():
    dists = compare_distributions(x_50=250.0, n_uniformity=1.25, b_swebrec=1.2, x_max=1000.0)
    assert "rosin_rammler" in dists
    assert "swebrec" in dists
    assert "lognormal" in dists

    fig = plot_distribution_comparison(dists)
    assert fig is not None


def test_calculate_goodness_of_fit():
    df1 = pd.DataFrame({"percent_passing": [10, 25, 50, 75, 90]})
    df2 = pd.DataFrame({"percent_passing": [12, 24, 51, 73, 91]})

    fit = calculate_goodness_of_fit(df1, df2)
    assert "r_squared" in fit
    assert "rmse" in fit
    assert "mae" in fit
    assert fit["r_squared"] > 0.95
