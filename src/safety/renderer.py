"""
Plotly interactive visualizations for safety and environmental modeling.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List


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
