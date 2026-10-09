"""
3D timing sequence visualization.
"""
import plotly.graph_objects as go
from src.render3d.hole_pattern import HolePattern


def timing_color_scale(delay_ms: float, max_delay_ms: float) -> str:
    """
    Map a delay value to a hex color (blue = early, red = late).
    """
    if max_delay_ms <= 0:
        return "#0000ff"
    t = min(1.0, max(0.0, delay_ms / max_delay_ms))
    r = int(255 * t)
    b = int(255 * (1 - t))
    return f"rgb({r},0,{b})"


def add_timing_trace(fig: go.Figure, pattern: HolePattern) -> None:
    """
    Add a Scatter3d trace for holes, color-coded by delay.
    """
    if not pattern.holes:
        raise ValueError("Pattern has no holes.")

    max_delay = max(h.delay_ms for h in pattern.holes)
    colors = [timing_color_scale(h.delay_ms, max_delay) for h in pattern.holes]
    xs = [h.x_m for h in pattern.holes]
    ys = [h.y_m for h in pattern.holes]
    zs = [h.z_collar_m for h in pattern.holes]
    hover = [
        f"{h.hole_id}<br>Delay: {h.delay_ms:.1f} ms<br>"
        f"Charge: {h.charge_kg:.1f} kg"
        for h in pattern.holes
    ]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=6, color=colors, line=dict(color="black", width=1)),
        text=hover,
        hoverinfo="text",
        name="Blastholes",
    ))


def add_timing_legend(fig: go.Figure, max_delay_ms: float) -> None:
    """
    Add a horizontal colorbar legend showing the delay scale.
    """
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=[[0, "#0000ff"], [1, "#ff0000"]],
            cmin=0, cmax=max_delay_ms,
            colorbar=dict(title="Delay (ms)", x=1.05),
            size=0,
        ),
        showlegend=False,
        hoverinfo="none",
    ))
