"""
3D timing animation and visual rendering for BlasterOPT 3D engine.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from src.render3d.hole_pattern import Hole


def create_timing_animation(
    holes: List[Hole],
    timing_sequence: Dict[str, Any],
    duration_ms: float = 2000.0,
) -> go.Figure:
    """
    Create an interactive 3D animation of the blast firing sequence.

    Each hole is a sphere/scatter point. Color encodes the delay time:
    - Blue: early (0 ms)
    - Yellow: middle
    - Red: late (max delay)

    Includes a frame slider that steps through time in 10 ms increments.
    At each frame, holes that have fired turn grey; holes that are about to
    fire pulse.
    """
    if not holes:
        fig = go.Figure()
        fig.update_layout(title="No holes in blast pattern")
        return fig

    # Map hole IDs to delay_ms
    delays_map = timing_sequence.get("delays_ms", {})
    hole_delays = []
    for h in holes:
        d = delays_map.get(h.hole_id, 0.0)
        hole_delays.append(d)

    max_delay = max(hole_delays) if max(hole_delays) > 0 else 1.0

    # Base coordinates
    x_coords = [h.x for h in holes]
    y_coords = [h.y for h in holes]
    z_coords = [h.z for h in holes]

    # Create initial 3D scatter trace
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers",
                marker=dict(
                    size=8,
                    color=hole_delays,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=max_delay,
                    colorbar=dict(title="Delay (ms)"),
                ),
                text=[f"{h.hole_id} ({d:.0f}ms)" for h, d in zip(holes, hole_delays)],
            )
        ]
    )

    # Frame animation in 10 ms increments
    frames = []
    time_steps = np.arange(0.0, duration_ms + 10.0, 10.0)

    for t in time_steps:
        frame_colors = []
        frame_sizes = []
        for d in hole_delays:
            if t >= d + 20.0:
                frame_colors.append("gray")  # Fired
                frame_sizes.append(6)
            elif abs(t - d) <= 20.0:
                frame_colors.append("red")  # Pulsing / firing
                frame_sizes.append(12)
            else:
                frame_colors.append("blue")  # Unfired / waiting
                frame_sizes.append(8)

        frame_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            marker=dict(
                size=frame_sizes,
                color=frame_colors,
            ),
            text=[f"{h.hole_id} (t={t:.0f}ms)" for h in holes],
        )
        frames.append(go.Frame(data=[frame_trace], name=f"frame_{int(t)}"))

    fig.frames = frames

    # Play/Pause controls and slider
    fig.update_layout(
        title="3D Blast Timing Firing Sequence",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Elevation (m)",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        label=f"{int(t)}ms",
                        args=[[f"frame_{int(t)}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                    )
                    for t in time_steps
                ],
                transition=dict(duration=0),
                x=0.1,
                len=0.9,
            )
        ],
    )

    return fig


def add_timing_to_plotter(plotter: Any, holes: List[Hole], timing_sequence: Dict[str, Any]) -> None:
    """Add timing colors to a PyVista plotter."""
    if plotter is not None and hasattr(plotter, "add_mesh"):
        import pyvista as pv

        delays_map = timing_sequence.get("delays_ms", {})
        coords = np.array([[h.x, h.y, h.z] for h in holes])
        delays = np.array([delays_map.get(h.hole_id, 0.0) for h in holes])

        cloud = pv.PolyData(coords)
        cloud["delay_ms"] = delays
        plotter.add_mesh(cloud, scalars="delay_ms", cmap="plasma", point_size=10, render_points_as_spheres=True)
