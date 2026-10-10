"""
Main 3D renderer using Plotly.

Design Choice Note:
===================
Plotly was explicitly chosen as the rendering engine for the 3D Blast Design module
because it is browser-native, requires zero C/C++ or system-level OpenGL/VTK dependencies,
and seamlessly integrates into Streamlit via `st.plotly_chart`.
"""
import numpy as np
import plotly.graph_objects as go
from src.render3d.bench_model import BenchGeometry, build_bench_mesh
from src.render3d.hole_pattern import HolePattern
from src.render3d.timing_viz import timing_color_scale


def render_bench_and_holes(
    bench: BenchGeometry,
    pattern: HolePattern,
    title: str = "3D Blast Design",
) -> go.Figure:
    """
    Return an interactive Plotly Figure showing the bench and drill holes.
    Enhanced with rock texture, vertical bench face strata lines, cylindrical holes,
    charge-weight color coding, North arrow, and compact horizontal colorbar.
    """
    mesh = build_bench_mesh(bench)
    fig = go.Figure()

    # 1. Textured Bench Crest Surface (Rock-colored material matrix)
    z_crest = mesh["crest"]["z"]
    # Subtle elevation noise for natural rock texture
    np.random.seed(42)
    noise = np.random.uniform(-0.15, 0.15, size=z_crest.shape)
    z_textured = z_crest + noise

    fig.add_trace(go.Surface(
        x=mesh["crest"]["x"],
        y=mesh["crest"]["y"],
        z=z_textured,
        colorscale=[[0, "#6E5B49"], [0.5, "#8B7355"], [1, "#A0896C"]],
        showscale=False,
        name="Bench Crest",
        opacity=0.92,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9),
    ))

    # 2. Vertical Bench Face with Geological Strata Lines
    fig.add_trace(go.Surface(
        x=mesh["face"]["x"],
        y=mesh["face"]["y"],
        z=mesh["face"]["z"],
        colorscale=[[0, "#4A3B32"], [0.33, "#705D4F"], [0.66, "#8B7355"], [1, "#5C4A3E"]],
        showscale=False,
        name="Bench Face",
        opacity=0.95,
    ))

    # Add 3 Strata Lines across the face
    length_m = bench.length_m
    height_m = bench.crest_elevation_m - bench.toe_elevation_m
    for idx, elev_ratio in enumerate([0.25, 0.50, 0.75]):
        z_strata = bench.toe_elevation_m + height_m * elev_ratio
        x_strata = np.linspace(0, length_m, 50)
        y_strata = np.zeros_like(x_strata)
        z_strata_arr = np.full_like(x_strata, z_strata)

        fig.add_trace(go.Scatter3d(
            x=x_strata,
            y=y_strata,
            z=z_strata_arr,
            mode="lines",
            line=dict(color="#2C221E", width=3),
            name=f"Strata Horizon {idx + 1}",
            showlegend=False,
            hoverinfo="name",
        ))

    # 3. Cylindrical Blastholes Color-coded by Charge Weight
    if pattern.holes:
        charges = [h.charge_kg for h in pattern.holes]
        min_c, max_c = min(charges), max(charges)
        if max_c == min_c:
            max_c = min_c + 1.0

        for h in pattern.holes:
            # Render hole trajectory as thick cylinder line
            norm_c = (h.charge_kg - min_c) / (max_c - min_c)
            # Viridis or Turbo charge color map
            color_hex = f"rgb({int(255 * norm_c)}, {int(180 * (1 - norm_c))}, {int(255 * (1 - norm_c))})"

            fig.add_trace(go.Scatter3d(
                x=[h.x_m, h.x_m],
                y=[h.y_m, h.y_m],
                z=[h.z_collar_m, h.z_toe_m],
                mode="lines+markers",
                line=dict(color=color_hex, width=8),
                marker=dict(size=[6, 4], color=color_hex),
                name=f"Hole {h.hole_id}",
                text=f"<b>{h.hole_id}</b><br>Charge: {h.charge_kg:.1f} kg<br>Delay: {h.delay_ms:.0f} ms<br>Depth: {h.depth_m:.1f} m",
                hoverinfo="text",
                showlegend=False,
            ))

        # Compact Horizontal Colorbar for Charge Weight
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(
                colorscale="Viridis",
                cmin=min_c,
                cmax=max_c,
                colorbar=dict(
                    title="Charge Weight (kg)",
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.05,
                    len=0.6,
                    thickness=12,
                    title_font=dict(size=12, color="#00B4D8"),
                    tickfont=dict(size=10),
                ),
                size=0,
            ),
            showlegend=False,
            hoverinfo="none",
        ))

    # 4. North Arrow Indicator
    max_x = max(bench.length_m, 20.0)
    max_y = max(bench.width_m, 20.0)
    arrow_x = [max_x * 0.9, max_x * 0.9]
    arrow_y = [max_y * 0.8, max_y * 0.95]
    arrow_z = [bench.crest_elevation_m, bench.crest_elevation_m]

    fig.add_trace(go.Scatter3d(
        x=arrow_x, y=arrow_y, z=arrow_z,
        mode="lines+text",
        line=dict(color="#00B4D8", width=5),
        text=["", "<b>N ⬆</b>"],
        textposition="top center",
        textfont=dict(color="#00B4D8", size=14),
        name="North Arrow",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#00B4D8")),
        scene=dict(
            xaxis=dict(title="X (m)", gridcolor="#333", backgroundcolor="#1E1E1E"),
            yaxis=dict(title="Y (m)", gridcolor="#333", backgroundcolor="#1E1E1E"),
            zaxis=dict(title="Z (m)", gridcolor="#333", backgroundcolor="#1E1E1E"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=580,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig
