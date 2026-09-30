"""
Multiple size distribution functions comparison (Rosin-Rammler, Swebrec, Log-Normal) and goodness of fit calculations.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Union
from src.fragmentation.swebrec import SwebrecModel


def rosin_rammler_curve(
    x_50: float,
    n: float,
    x_min: float = 0.1,
    x_max: float = 2000.0,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Rosin-Rammler distribution (used in Kuz-Ram).

    P(x) = 100 * (1 - exp(-0.693 * (x / x_50)^n))

    Reference: Rosin, P., & Rammler, E. (1933). "The laws governing the
    fineness of powdered coal." Journal of the Institute of Fuel, 7, 29-36.
    """
    if x_50 <= 0 or n <= 0:
        raise ValueError("x_50 and n must be strictly positive")

    sizes = np.logspace(np.log10(max(0.01, x_min)), np.log10(x_max), n_points)
    # Formula: P(x) = 100 * (1 - exp(-0.693 * (x / x_50)^n))
    p_vals = 100.0 * (1.0 - np.exp(-0.693 * ((sizes / x_50) ** n)))

    return pd.DataFrame({
        "size_mm": sizes,
        "percent_passing": np.clip(p_vals, 0.0, 100.0),
    })


def lognormal_curve(
    x_50: float,
    sigma: float,
    x_min: float = 0.1,
    x_max: float = 2000.0,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Log-normal distribution.

    P(x) = 100 * 0.5 * (1 + erf((ln(x) - mu) / (sigma * sqrt(2))))
    where mu = ln(x_50).
    """
    if x_50 <= 0 or sigma <= 0:
        raise ValueError("x_50 and sigma must be strictly positive")

    sizes = np.logspace(np.log10(max(0.01, x_min)), np.log10(x_max), n_points)
    mu = math.log(x_50)

    # Formula: P(x) = 100 * 0.5 * (1 + erf((ln(x) - mu) / (sigma * sqrt(2))))
    erf_vec = np.vectorize(math.erf)
    p_vals = 100.0 * 0.5 * (1.0 + erf_vec((np.log(sizes) - mu) / (sigma * math.sqrt(2.0))))

    return pd.DataFrame({
        "size_mm": sizes,
        "percent_passing": np.clip(p_vals, 0.0, 100.0),
    })


def compare_distributions(
    x_50: float,
    n_uniformity: float,
    b_swebrec: float,
    x_max: float = 2000.0,
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict with three DataFrames: 'rosin_rammler', 'swebrec', 'lognormal'.
    Each has columns 'size_mm' and 'percent_passing'.
    """
    df_rr = rosin_rammler_curve(x_50=x_50, n=n_uniformity, x_min=0.1, x_max=x_max)

    swe_mod = SwebrecModel(x_max=x_max, x_50=x_50, b=b_swebrec)
    df_sw = swe_mod.compute_curve(x_min=0.1, x_max=x_max, n_points=200)

    df_ln = lognormal_curve(x_50=x_50, sigma=0.6, x_min=0.1, x_max=x_max)

    return {
        "rosin_rammler": df_rr,
        "swebrec": df_sw,
        "lognormal": df_ln,
    }


def plot_distribution_comparison(distributions: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Creates a Plotly figure overlaying all three distributions.
    X-axis: fragment size (log scale)
    Y-axis: percent passing
    """
    fig = go.Figure()

    if "swebrec" in distributions:
        df = distributions["swebrec"]
        fig.add_trace(go.Scatter(x=df["size_mm"], y=df["percent_passing"], mode="lines", name="Swebrec (Ouchterlony 2005)", line=dict(color="#2962FF", width=3)))

    if "rosin_rammler" in distributions:
        df = distributions["rosin_rammler"]
        fig.add_trace(go.Scatter(x=df["size_mm"], y=df["percent_passing"], mode="lines", name="Rosin-Rammler (Rosin & Rammler 1933)", line=dict(color="#FF6D00", width=2, dash="dash")))

    if "lognormal" in distributions:
        df = distributions["lognormal"]
        fig.add_trace(go.Scatter(x=df["size_mm"], y=df["percent_passing"], mode="lines", name="Log-Normal", line=dict(color="#00C853", width=2, dash="dot")))

    fig.update_layout(
        title="<b>Fragment Size Distribution Functions Comparison</b>",
        xaxis_title="Fragment Size x (mm)",
        yaxis_title="Cumulative Passing P(x) (%)",
        xaxis_type="log",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
    )

    return fig


def calculate_goodness_of_fit(measured: pd.DataFrame, predicted: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates R², RMSE, and MAE between measured and predicted curves.
    Returns {'r_squared': float, 'rmse': float, 'mae': float}.
    """
    y_meas = measured["percent_passing"].values
    y_pred = predicted["percent_passing"].values

    if len(y_meas) != len(y_pred):
        min_len = min(len(y_meas), len(y_pred))
        y_meas = y_meas[:min_len]
        y_pred = y_pred[:min_len]

    if len(y_meas) == 0:
        raise ValueError("Measured array cannot be empty")

    residuals = y_meas - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_meas - np.mean(y_meas))**2))
    r_squared = float(1.0 - (ss_res / max(1e-5, ss_tot)))

    return {
        "r_squared": round(r_squared, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }
