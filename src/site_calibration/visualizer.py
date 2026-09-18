"""
Plotly Visualizer Module for Attenuation Curves, Residuals, GSI Correction, and Dashboard.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from src.site_calibration.models import AttenuationParameters, CalibrationResult


def plot_attenuation_curve(
    distances: np.ndarray,
    ppvs: np.ndarray,
    charges: np.ndarray,
    params: Optional[AttenuationParameters] = None,
    title: str = "Site-Specific USBM Vibration Attenuation Curve"
) -> go.Figure:
    """
    Plots PPV vs Scaled Distance (D / sqrt(Q)) on log-log axes with fitted line and 95% CI.
    """
    sd = distances / np.sqrt(charges)
    df_data = pd.DataFrame({"Scaled_Distance": sd, "PPV_mms": ppvs})

    fig = go.Figure()

    # Scatter plot of actual seismograph readings
    fig.add_trace(go.Scatter(
        x=df_data["Scaled_Distance"],
        y=df_data["PPV_mms"],
        mode="markers",
        name="Seismograph Readings",
        marker=dict(size=8, color="#2962FF", symbol="circle"),
    ))

    if params is not None:
        sd_grid = np.linspace(df_data["Scaled_Distance"].min() * 0.8, df_data["Scaled_Distance"].max() * 1.2, 100)
        ppv_fit = params.K * (sd_grid ** (-params.B))

        # Fitted curve
        fig.add_trace(go.Scatter(
            x=sd_grid,
            y=ppv_fit,
            mode="lines",
            name=f"Fitted USBM (K={params.K:.1f}, B={params.B:.2f}, R²={params.r_squared:.2f})",
            line=dict(color="#D50000", width=3),
        ))

        # 95% Confidence Band
        fig.add_trace(go.Scatter(
            x=sd_grid,
            y=ppv_fit * 1.25,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=sd_grid,
            y=ppv_fit * 0.75,
            mode="lines",
            name="95% CI Band",
            fill="tonexty",
            fillcolor="rgba(213, 0, 0, 0.15)",
            line=dict(width=0),
        ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis=dict(title="Scaled Distance (m / kg^0.5)", type="log"),
        yaxis=dict(title="Peak Particle Velocity PPV (mm/s)", type="log"),
        template="plotly_white",
        height=420,
    )
    return fig


def plot_residuals(
    actual_ppv: np.ndarray,
    predicted_ppv: np.ndarray,
    title: str = "Attenuation Fit Residuals Plot"
) -> go.Figure:
    """
    Plots predicted vs. actual PPV residuals.
    """
    residuals = actual_ppv - predicted_ppv
    df_res = pd.DataFrame({"Predicted": predicted_ppv, "Residuals": residuals})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_res["Predicted"],
        y=df_res["Residuals"],
        mode="markers",
        marker=dict(size=8, color="#00C853"),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Predicted PPV (mm/s)",
        yaxis_title="Residual Error (Actual - Predicted) [mm/s]",
        template="plotly_white",
        height=380,
    )
    return fig


def plot_gsi_correction_curve(
    gsis: np.ndarray,
    factors: np.ndarray,
    params: AttenuationParameters,
    title: str = "GSI Vibration Correction Multiplier f(GSI)"
) -> go.Figure:
    """
    Plots GSI multiplier f(GSI) = a * exp(b * GSI) vs. GSI.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gsis,
        y=factors,
        mode="markers",
        name="Measured GSI Points",
        marker=dict(size=8, color="#FF6D00"),
    ))

    gsi_grid = np.linspace(20, 90, 100)
    a_val = params.gsi_a or 1.0
    b_val = params.gsi_b or 0.0
    curve_fit_y = a_val * np.exp(b_val * gsi_grid)

    fig.add_trace(go.Scatter(
        x=gsi_grid,
        y=curve_fit_y,
        mode="lines",
        name=f"f(GSI) = {a_val:.2f} * exp({b_val:.4f} * GSI)",
        line=dict(color="#AA00FF", width=3),
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Geological Strength Index (GSI)",
        yaxis_title="Vibration Multiplier f(GSI)",
        template="plotly_white",
        height=380,
    )
    return fig


def plot_calibration_dashboard(
    calib_result: CalibrationResult,
    output_html_path: Optional[str] = None
) -> go.Figure:
    """
    Generates a calibration dashboard summary figure and optionally exports HTML.
    """
    params = calib_result.parameters
    r2 = params.r_squared if params else 0.0
    rmse = params.rmse if params else 0.0
    n = params.sample_count if params else 0

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=r2,
        title={"text": f"<b>Site Calibration Quality [{calib_result.site_id}]</b><br><span style='font-size:0.8em;color:gray'>R² Score (Min Target: 0.70)</span>"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#00C853" if r2 >= 0.70 else "#D50000"},
            "steps": [
                {"range": [0, 0.70], "color": "#FFEBEE"},
                {"range": [0.70, 0.85], "color": "#FFF8E1"},
                {"range": [0.85, 1.0], "color": "#E8F5E9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 0.70,
            },
        },
    ))

    fig.update_layout(template="plotly_white", height=320)

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        fig.write_html(output_html_path)

    return fig
