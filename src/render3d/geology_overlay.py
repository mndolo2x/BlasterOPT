"""
Geological overlay for the 3D bench scene.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Tuple


def add_rock_layer(
    fig: go.Figure,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_top: float,
    z_bottom: float,
    color: str,
    name: str,
) -> None:
    """
    Add a rock layer as a semi-transparent 3D box.
    """
    x0, x1 = x_range
    y0, y1 = y_range

    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z_bottom, z_bottom, z_bottom, z_bottom, z_top, z_top, z_top, z_top]

    i = [0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 1, 1]
    j = [1, 2, 3, 1, 5, 6, 7, 5, 4, 5, 2, 6]
    k = [2, 3, 0, 5, 6, 7, 4, 1, 5, 1, 6, 2]

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=0.25,
        name=name,
        hoverinfo="name",
    ))


def add_fault_plane(
    fig: go.Figure,
    polyline: np.ndarray,
    dip_deg: float,
    color: str = "#ff0000",
) -> None:
    """
    Add a fault as a 3D plane surface.
    """
    if polyline.shape[1] != 3:
        raise ValueError("Polyline must be Nx3 array.")

    dip_rad = np.radians(dip_deg)
    extrusion = np.array([
        [0, np.cos(dip_rad), -np.sin(dip_rad)],
        [0, np.cos(dip_rad), -np.sin(dip_rad)],
    ]) * 20.0

    x = np.concatenate([polyline[:, 0], polyline[:, 0] + extrusion[0, 0]])
    y = np.concatenate([polyline[:, 1], polyline[:, 1] + extrusion[0, 1]])
    z = np.concatenate([polyline[:, 2], polyline[:, 2] + extrusion[0, 2]])

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=4),
        name="Fault",
    ))
