"""
Main 3D renderer using Plotly.

Design Choice Note:
===================
Plotly was explicitly chosen as the rendering engine for the 3D Blast Design module
because it is browser-native, requires zero C/C++ or system-level OpenGL/VTK dependencies,
and seamlessly integrates into Streamlit via `st.plotly_chart`.
"""
import plotly.graph_objects as go
from src.render3d.bench_model import BenchGeometry, build_bench_mesh
from src.render3d.hole_pattern import HolePattern
from src.render3d.timing_viz import add_timing_trace, add_timing_legend


def render_bench_and_holes(
    bench: BenchGeometry,
    pattern: HolePattern,
    title: str = "3D Blast Design",
) -> go.Figure:
    """
    Return an interactive Plotly Figure showing the bench and drill holes.
    """
    mesh = build_bench_mesh(bench)
    fig = go.Figure()

    # Crest surface
    fig.add_trace(go.Surface(
        x=mesh["crest"]["x"],
        y=mesh["crest"]["y"],
        z=mesh["crest"]["z"],
        colorscale=[[0, "#8B7355"], [1, "#8B7355"]],
        showscale=False,
        name="Bench crest",
        opacity=0.9,
    ))

    # Face surface
    fig.add_trace(go.Surface(
        x=mesh["face"]["x"],
        y=mesh["face"]["y"],
        z=mesh["face"]["z"],
        colorscale=[[0, "#A0896C"], [1, "#A0896C"]],
        showscale=False,
        name="Bench face",
        opacity=0.95,
    ))

    # Drill holes
    add_timing_trace(fig, pattern)
    if pattern.holes:
        add_timing_legend(fig, max(h.delay_ms for h in pattern.holes))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig
