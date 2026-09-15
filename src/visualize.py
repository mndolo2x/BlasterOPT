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
    d50_mm: float, n_uniformity: float = 1.2, label: str = "Predicted Blast Pattern"
) -> go.Figure:
    """
    Plots Kuz-Ram / Rosin-Rammler cumulative passing size distribution curve.

    Rosin-Rammler Equation:
    P(x) = 1 - exp( - (x / x_c)^n )
    where x_c = d50 / (ln(2))^(1/n)
    """
    x_sizes = np.logspace(0, 3.2, 200) # 1 mm to 1500 mm

    # Characteristic size x_c
    x_c = d50_mm / (np.log(2.0) ** (1.0 / max(n_uniformity, 0.1)))

    # Passing percentage P(x) in %
    passing_pct = (1.0 - np.exp(-1.0 * ((x_sizes / x_c) ** n_uniformity))) * 100.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_sizes,
            y=passing_pct,
            mode="lines",
            name=f"{label} (d50={d50_mm:.1f} mm, n={n_uniformity:.2f})",
            line=dict(color="#2962FF", width=3),
            hovertemplate="Size: %{x:.1f} mm<br>Passing: %{y:.1f}%<extra></extra>",
        )
    )

    # Reference threshold lines (e.g., d50 and oversize > 500mm)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Passing (d50)")
    fig.add_vline(x=d50_mm, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"<b>Kuz-Ram Size Distribution Curve</b>",
        xaxis_title="Sieve Size / Fragment Size (mm) [Log Scale]",
        yaxis_title="Cumulative Passing (%)",
        xaxis_type="log",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        height=450,
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
