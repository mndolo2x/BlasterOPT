"""
As-drilled vs as-designed comparison.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
from src.render3d.hole_pattern import HolePattern


def compare_designs(
    designed: HolePattern, as_drilled: HolePattern
) -> Dict[str, Any]:
    """
    Compare as-designed and as-drilled patterns.

    Returns a dict with:
        mean_xy_offset_m: average horizontal deviation
        max_xy_offset_m: worst horizontal deviation
        mean_depth_diff_m: average depth difference
        worst_hole_id: hole with the largest offset
    """
    if len(designed.holes) != len(as_drilled.holes):
        raise ValueError("Both patterns must have the same number of holes.")

    offsets = []
    depth_diffs = []
    worst = (0.0, None)

    for d, a in zip(designed.holes, as_drilled.holes):
        dx = a.x_m - d.x_m
        dy = a.y_m - d.y_m
        offset = float(np.sqrt(dx**2 + dy**2))
        offsets.append(offset)
        depth_diffs.append(a.depth_m - d.depth_m)
        if offset >= worst[0]:
            worst = (offset, d.hole_id)

    return {
        "mean_xy_offset_m": float(np.mean(offsets)),
        "max_xy_offset_m": float(np.max(offsets)),
        "mean_depth_diff_m": float(np.mean(depth_diffs)),
        "worst_hole_id": worst[1],
    }


def add_as_drilled_trace(fig: go.Figure, as_drilled: HolePattern) -> None:
    """Overlay as-drilled holes on the 3D figure as hollow circles."""
    fig.add_trace(go.Scatter3d(
        x=[h.x_m for h in as_drilled.holes],
        y=[h.y_m for h in as_drilled.holes],
        z=[h.z_collar_m for h in as_drilled.holes],
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(color="green", width=2)),
        name="As-drilled",
    ))
