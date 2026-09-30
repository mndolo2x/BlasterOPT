"""
Plotly interactive visualizations for safety and environmental modeling.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Union


class SafetyRenderer:
    """Plotly interactive renderer for safety, dispersion, and risk heatmaps."""

    def plot_dust_plume(self, pm10_concentration_ug_m3: float, distance_m: float) -> go.Figure:
        """Render Plotly 2D/3D downwind plume dispersion profile."""
        x_dist = np.linspace(10, max(500.0, distance_m * 1.5), 100)
        # Plume decay profile
        conc = pm10_concentration_ug_m3 * (distance_m / x_dist) ** 1.5

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_dist,
                y=conc,
                mode="lines",
                name="Downwind PM10 (ug/m3)",
                line=dict(color="#FF6D00", width=3),
            )
        )
        fig.add_hline(y=150.0, line_dash="dash", line_color="red", annotation_text="WHO 24-hr Limit (150 ug/m3)")

        fig.update_layout(
            title="<b>Downwind Dust (PM10) Gaussian Plume Dispersion</b>",
            xaxis_title="Downwind Distance x (m)",
            yaxis_title="PM10 Concentration (ug/m3)",
            template="plotly_white",
        )
        return fig


def plot_dust_dispersion(dust_model: Any, distances: np.ndarray) -> go.Figure:
    """
    Plots dust concentration vs. distance.
    """
    concentrations = []
    for d in distances:
        try:
            res = dust_model.calculate_pm10_concentration(dust_mass_kg=100.0, distance_m=float(d), wind_speed_m_s=3.0)
            concentrations.append(res["pm10_concentration_mg_m3"])
        except Exception:
            concentrations.append(0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=concentrations,
            mode="lines+markers",
            name="PM10 Concentration (mg/m³)",
            line=dict(color="#2E7D32", width=3),
        )
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="WHO 24-hr Threshold (0.05 mg/m³)")

    fig.update_layout(
        title="<b>Dust PM10 Concentration vs. Distance</b>",
        xaxis_title="Distance from Blast (m)",
        yaxis_title="PM10 Concentration (mg/m³)",
        template="plotly_white",
    )
    return fig


def plot_gas_concentration(gas_model: Any, times: np.ndarray) -> go.Figure:
    """
    Plots gas concentration vs. time.
    """
    concentrations = []
    for t in times:
        try:
            res = gas_model.calculate_gas_concentration(
                gas_mass_kg=15.0,
                gas_type="CO",
                ventilation_rate_m3_s=10.0,
                volume_m3=5000.0,
                time_s=float(t),
            )
            concentrations.append(res["concentration_ppm"])
        except Exception:
            concentrations.append(0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=concentrations,
            mode="lines",
            name="CO Concentration (ppm)",
            line=dict(color="#D32F2F", width=3),
        )
    )
    fig.add_hline(y=50.0, line_dash="dash", line_color="orange", annotation_text="CO 8-hr TWA Limit (50 ppm)")

    fig.update_layout(
        title="<b>Toxic Gas (CO) Concentration vs. Ventilation Time</b>",
        xaxis_title="Time Post-Blast (s)",
        yaxis_title="Concentration (ppm)",
        template="plotly_white",
    )
    return fig


def plot_noise_attenuation(noise_model: Any, distances: np.ndarray) -> go.Figure:
    """
    Plots noise level vs. distance.
    """
    spl_levels = []
    for d in distances:
        try:
            res = noise_model.calculate_noise_at_distance(float(d))
            spl_levels.append(res["spl_db"])
        except Exception:
            spl_levels.append(0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=spl_levels,
            mode="lines+markers",
            name="Sound Pressure Level (dB)",
            line=dict(color="#0288D1", width=3),
        )
    )
    fig.add_hline(y=120.0, line_dash="dash", line_color="red", annotation_text="Botswana Department of Mines Limit (120 dBL)")

    fig.update_layout(
        title="<b>Airblast Noise Attenuation vs. Distance</b>",
        xaxis_title="Distance to Receptor (m)",
        yaxis_title="Sound Pressure Level (dBL)",
        template="plotly_white",
    )
    return fig


def plot_risk_matrix(risk_analyzer: Any) -> go.Figure:
    """
    Plots the 5x5 risk matrix with all risks.
    """
    params = {"closest_receptor_dist_m": 500.0}
    return risk_analyzer.generate_risk_matrix_plot(params)


def plot_environmental_breakdown(eia: Any) -> go.Figure:
    """
    Plots the environmental impact breakdown as a radar chart.
    """
    try:
        impact = eia.calculate_total_impact()
        bd = impact["breakdown"]
    except Exception:
        bd = {"dust": 2, "gas": 2, "noise": 3, "vibration": 3, "flyrock": 2}

    categories = ["Dust", "Toxic Gas", "Noise", "Vibration", "Flyrock"]
    scores = [bd.get("dust", 1), bd.get("gas", 1), bd.get("noise", 1), bd.get("vibration", 1), bd.get("flyrock", 1)]

    # Close radar loop
    categories_loop = categories + [categories[0]]
    scores_loop = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores_loop,
            theta=categories_loop,
            fill="toself",
            name="EIA Impact Scores",
            line=dict(color="#7B1FA2", width=3),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1 (Negligible)", "2 (Minor)", "3 (Moderate)", "4 (Major)", "5 (Severe)"],
            )
        ),
        title="<b>Environmental Impact Assessment Category Breakdown (1-5)</b>",
        template="plotly_white",
    )
    return fig
