"""
Visualization Module for BlastOpt Botswana.

Generates interactive Plotly and Matplotlib figures for:
- Kuz-Ram cumulative size distribution curves (Rosin-Rammler)
- USBM Ground Vibration attenuation curves
- Feature Importance charts
- GA Optimization convergence plots
- 2D Blast Pattern geometry and initiation delay sequence layout
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple


def plot_kuz_ram_curve(
    d50_mm: float,
    n_uniformity: float = 1.2,
    xc_custom_mm: Optional[float] = None,
    label: str = "Predicted Blast Pattern",
) -> go.Figure:
    """
    Generates a Plotly interactive Kuz-Ram / Rosin-Rammler fragmentation size distribution curve.

    Mining Engineering Context & Mathematical Logic:
    ------------------------------------------------
    The Kuz-Ram model is an established empirical fragmentation model in mining engineering used
    to estimate the particle size distribution of blasted rock mass. It combines Cunningham's
    extended Kuz-Ram equation for mean fragment size (d50) with the Rosin-Rammler cumulative
    distribution function.

    Rosin-Rammler Cumulative Passing Formula:
        P(x) = 1 - exp( - (x / x_c)^n )

    Where:
    - P(x): Cumulative fraction of rock mass passing through a sieve of aperture size x (in %).
    - x: Sieve aperture size or fragment dimension (mm).
    - x_c: Characteristic particle size parameter (mm), defined as the sieve aperture size
           through which 63.2% (1 - 1/e) of the blasted rock mass passes.
           Mathematically related to d50 (50% passing size) by:
               x_c = d50 / (ln(2))^(1 / n)
    - n: Uniformity index (typically 0.8 - 2.2 for rock blasting). Higher values of n indicate
         a tightly grouped, uniform size distribution with fewer boulders or fines.

    Parameters:
    -----------
    d50_mm : float
        Mean fragment size (50% passing diameter) in millimeters.
    n_uniformity : float, default=1.2
        Rosin-Rammler uniformity index (n). Controls the slope of the passing curve.
    xc_custom_mm : float, optional
        Custom characteristic size x_c parameter in millimeters. If provided, overrides
        the d50-derived characteristic size calculation.
    label : str, default="Predicted Blast Pattern"
        Legend label for the trace.

    Returns:
    --------
    go.Figure
        Plotly Figure displaying the logarithmic cumulative size distribution curve.
    """
    x_sizes = np.logspace(0, 3.2, 250)  # Sieve size range from 1 mm to 1585 mm

    # Derive or assign characteristic size x_c
    n_val = max(float(n_uniformity), 0.1)
    if xc_custom_mm is not None and xc_custom_mm > 0:
        x_c = float(xc_custom_mm)
    else:
        # x_c derived from d50: x_c = d50 / (ln 2)^(1/n)
        x_c = d50_mm / (np.log(2.0) ** (1.0 / n_val))

    # Calculate cumulative passing percentage P(x)
    passing_pct = (1.0 - np.exp(-1.0 * ((x_sizes / x_c) ** n_val))) * 100.0

    # Calculated d50 for reference annotation
    effective_d50 = x_c * (np.log(2.0) ** (1.0 / n_val))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_sizes,
            y=passing_pct,
            mode="lines",
            name=f"{label} (x_c={x_c:.1f} mm, n={n_val:.2f})",
            line=dict(color="#2962FF", width=3),
            hovertemplate="Fragment Size: %{x:.1f} mm<br>Cumulative Passing: %{y:.1f}%<extra></extra>",
        )
    )

    # Reference threshold indicator lines
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"d50 = {effective_d50:.1f} mm",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=63.2,
        line_dash="dot",
        line_color="darkblue",
        annotation_text=f"x_c (63.2% Passing) = {x_c:.1f} mm",
        annotation_position="top right",
    )
    fig.add_vline(x=effective_d50, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="<b>Kuz-Ram Fragmentation Curve (Percent Passing vs Size)</b>",
        xaxis_title="Sieve / Fragment Size x (mm) [Log Scale]",
        yaxis_title="Cumulative Percent Passing P(x) [%]",
        xaxis_type="log",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        height=500,
    )

    return fig


def plot_ppv_attenuation(
    max_charge_kg: float, k_vib: float = 1140.0, beta: float = 1.6
) -> go.Figure:
    """
    Plots USBM scale distance ground vibration attenuation curve (PPV vs Distance).
    """
    distances = np.linspace(50, 1500, 200) # meters
    scaled_distances = distances / np.sqrt(max_charge_kg)
    ppv = k_vib * (scaled_distances ** (-beta))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=distances,
            y=ppv,
            mode="lines",
            name=f"Q = {max_charge_kg:.0f} kg/delay",
            line=dict(color="#D50000", width=3),
            hovertemplate="Distance: %{x:.0f} m<br>PPV: %{y:.2f} mm/s<extra></extra>",
        )
    )

    # Common compliance limits (e.g. 5 mm/s, 10 mm/s, 25 mm/s)
    fig.add_hline(y=10.0, line_dash="dot", line_color="orange", annotation_text="10 mm/s Standard Limit")
    fig.add_hline(y=5.0, line_dash="dot", line_color="green", annotation_text="5 mm/s Sensitive Limit")

    fig.update_layout(
        title=f"<b>Ground Vibration Attenuation (PPV vs Distance)</b>",
        xaxis_title="Distance to Target (m)",
        yaxis_title="Peak Particle Velocity (PPV) [mm/s]",
        template="plotly_white",
        height=450,
    )

    return fig


def plot_feature_importance(importance_df: pd.DataFrame, target: str) -> go.Figure:
    """
    Plots feature importance horizontal bar chart for a target variable.
    """
    if target not in importance_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No feature importances available", showarrow=False)
        return fig

    df_sorted = importance_df.sort_values(by=target, ascending=True)

    fig = px.bar(
        df_sorted,
        x=target,
        y="feature",
        orientation="h",
        title=f"<b>Feature Importance for {target}</b>",
        labels={target: "Relative Importance", "feature": "Feature"},
        color=target,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_white", height=450)
    return fig


def plot_optimization_convergence(history: List[float]) -> go.Figure:
    """
    Plots GA Optimization convergence curve (Objective cost penalty vs Iteration).
    """
    iterations = list(range(1, len(history) + 1))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=history,
            mode="lines+markers",
            name="Fitness Value",
            line=dict(color="#00C853", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="<b>Genetic Algorithm Convergence History</b>",
        xaxis_title="Evaluation Step / Iteration",
        yaxis_title="Objective Function Cost Score ($/t + penalties)",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_2d_blast_pattern(
    num_rows: int = 4,
    holes_per_row: int = 8,
    burden_m: float = 6.0,
    spacing_m: float = 7.0,
    row_delay_ms: int = 42,
    hole_delay_ms: int = 17,
) -> go.Figure:
    """
    Generates 2D Blast Hole Layout with initiation delay timing overlay.
    """
    x_coords = []
    y_coords = []
    delays = []
    hole_labels = []

    for r in range(num_rows):
        for h in range(holes_per_row):
            # Staggered or rectangular grid
            offset = (r % 2) * (spacing_m / 2.0)
            x = h * spacing_m + offset
            y = r * burden_m

            # Initiation delay time calculation (e.g. V-pattern or row-by-row)
            delay = r * row_delay_ms + h * hole_delay_ms

            x_coords.append(x)
            y_coords.append(y)
            delays.append(delay)
            hole_labels.append(f"R{r+1}-H{h+1}<br>{delay} ms")

    df_pattern = pd.DataFrame({
        "x": x_coords,
        "y": y_coords,
        "delay_ms": delays,
        "label": hole_labels,
    })

    fig = px.scatter(
        df_pattern,
        x="x",
        y="y",
        color="delay_ms",
        size_max=15,
        text="delay_ms",
        color_continuous_scale="Plasma",
        title="<b>2D Blast Hole Layout & Delay Sequence Simulator</b>",
        labels={"x": "Easting / Spacing Distance (m)", "y": "Northing / Burden Distance (m)", "delay_ms": "Delay (ms)"},
    )

    fig.update_traces(
        marker=dict(size=22, line=dict(width=1.5, color="black")),
        textposition="top center",
        hovertemplate="<b>Hole %{text}</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
    )

    fig.update_layout(
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=550,
    )

    return fig
