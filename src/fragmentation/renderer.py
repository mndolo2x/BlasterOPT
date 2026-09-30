"""
Plotly interactive visualizations for fragmentation models.
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.fragmentation.distributions import (
    compare_distributions,
    plot_distribution_comparison as base_plot_comparison,
)


def plot_fragmentation_curve(
    distribution_df: pd.DataFrame,
    model_name: str = "Kuz-Ram",
    measured_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Plots a single fragmentation curve.
    If measured_df is provided, overlays measured data points.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=distribution_df["size_mm"],
            y=distribution_df["percent_passing"],
            mode="lines",
            name=f"Predicted ({model_name})",
            line=dict(color="#2962FF", width=3),
        )
    )

    if measured_df is not None and not measured_df.empty:
        fig.add_trace(
            go.Scatter(
                x=measured_df["size_mm"],
                y=measured_df["percent_passing"],
                mode="markers",
                name="Measured (WipFrag / Split-Desktop)",
                marker=dict(size=8, color="#D50000", symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"<b>Fragmentation Size Distribution Curve — {model_name}</b>",
        xaxis_title="Particle Size x (mm)",
        yaxis_title="Cumulative Passing P(x) (%)",
        xaxis_type="log",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
    )

    return fig


def plot_model_comparison(
    distributions: dict, measured_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Plots Rosin-Rammler, Swebrec, and log-normal on one figure.
    """
    fig = base_plot_comparison(distributions)

    if measured_df is not None and not measured_df.empty:
        fig.add_trace(
            go.Scatter(
                x=measured_df["size_mm"],
                y=measured_df["percent_passing"],
                mode="markers",
                name="Measured Data Points",
                marker=dict(size=8, color="#D50000", symbol="x"),
            )
        )

    return fig


def plot_calibration_scatter(
    measured_d80: np.ndarray,
    predicted_d80: np.ndarray,
    calibrated_predicted: np.ndarray,
) -> go.Figure:
    """
    Scatter plot comparing measured vs. predicted (before and after calibration).
    Includes 1:1 reference line.
    """
    fig = go.Figure()

    # 1:1 Reference Line
    min_val = min(np.min(measured_d80), np.min(predicted_d80), np.min(calibrated_predicted)) * 0.9
    max_val = max(np.max(measured_d80), np.max(predicted_d80), np.max(calibrated_predicted)) * 1.1

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="1:1 Parity Line",
            line=dict(color="gray", dash="dash", width=2),
        )
    )

    # Uncalibrated
    fig.add_trace(
        go.Scatter(
            x=measured_d80,
            y=predicted_d80,
            mode="markers",
            name="Uncalibrated Kuz-Ram",
            marker=dict(size=10, color="#FF6D00", symbol="circle"),
        )
    )

    # Calibrated
    fig.add_trace(
        go.Scatter(
            x=measured_d80,
            y=calibrated_predicted,
            mode="markers",
            name="Calibrated RSM Model",
            marker=dict(size=10, color="#00C853", symbol="square"),
        )
    )

    fig.update_layout(
        title="<b>Rock-Specific Calibration Parity Plot (Measured vs. Predicted D80)</b>",
        xaxis_title="Measured D80 (cm)",
        yaxis_title="Predicted D80 (cm)",
        template="plotly_white",
    )

    return fig


class FragmentationRenderer:
    """Plotly interactive renderer for fragmentation size distribution curves."""

    def plot_distribution_comparison(
        self,
        d50_mm: float,
        n_uniformity: float = 1.25,
        b_swebrec: float = 1.25,
        x_max_mm: float = 1000.0,
    ) -> go.Figure:
        dists = compare_distributions(
            x_50=d50_mm,
            n_uniformity=n_uniformity,
            b_swebrec=b_swebrec,
            x_max=x_max_mm,
        )
        return plot_model_comparison(dists)
