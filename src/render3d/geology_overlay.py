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
    color: str = "#E53935",
    name: str = "FAULT_JWA_01",
) -> None:
    """
    Add a fault as a 3D translucent plane surface intersecting the bench.
    """
    if polyline.shape[1] != 3:
        raise ValueError("Polyline must be Nx3 array.")

    p1, p2 = polyline[0], polyline[1]
    dip_rad = np.radians(dip_deg)
    depth = 25.0

    # 4 corners of the fault plane
    x0, y0, z0 = p1
    x1, y1, z1 = p2

    x2 = x1
    y2 = y1 + depth * np.cos(dip_rad)
    z2 = z1 - depth * np.sin(dip_rad)

    x3 = x0
    y3 = y0 + depth * np.cos(dip_rad)
    z3 = z0 - depth * np.sin(dip_rad)

    x_mesh = [x0, x1, x2, x3]
    y_mesh = [y0, y1, y2, y3]
    z_mesh = [z0, z1, z2, z3]

    i = [0, 0]
    j = [1, 2]
    k = [2, 3]

    fig.add_trace(go.Mesh3d(
        x=x_mesh, y=y_mesh, z=z_mesh,
        i=i, j=j, k=k,
        color=color,
        opacity=0.45,
        name=name,
        hoverinfo="name",
    ))


def add_geological_overlays(
    fig: go.Figure,
    bench_length_m: float,
    bench_width_m: float,
    crest_elevation_m: float,
    toe_elevation_m: float,
    show_geology: bool = True,
    show_fault: bool = True,
) -> None:
    """
    Add 3D volumetric geological layers and fault plane overlay.
    - Kimberlite: translucent purple (#8E24AA)
    - Dolomite waste: translucent gray (#78909C)
    - Granite contact: translucent orange (#FB8C00) on eastern side
    - Fault FAULT_JWA_01: translucent red (#E53935)
    """
    bottom_z = toe_elevation_m - 2.0

    if show_geology:
        # 1. Kimberlite Pipe Core (Center)
        add_rock_layer(
            fig,
            x_range=(bench_length_m * 0.2, bench_length_m * 0.7),
            y_range=(bench_width_m * 0.1, bench_width_m * 0.8),
            z_top=crest_elevation_m,
            z_bottom=bottom_z,
            color="#8E24AA",
            name="Kimberlite Ore",
        )

        # 2. Dolomite Waste Rock (Western Side)
        add_rock_layer(
            fig,
            x_range=(0.0, bench_length_m * 0.2),
            y_range=(0.0, bench_width_m),
            z_top=crest_elevation_m,
            z_bottom=bottom_z,
            color="#78909C",
            name="Dolomite Waste",
        )

        # 3. Granite Wall Contact (Eastern Side)
        add_rock_layer(
            fig,
            x_range=(bench_length_m * 0.7, bench_length_m),
            y_range=(0.0, bench_width_m),
            z_top=crest_elevation_m,
            z_bottom=bottom_z,
            color="#FB8C00",
            name="Granite Contact",
        )

    if show_fault:
        # 4. Structural Fault Plane FAULT_JWA_01
        fault_polyline = np.array([
            [bench_length_m * 0.1, bench_width_m * 0.2, crest_elevation_m],
            [bench_length_m * 0.8, bench_width_m * 0.9, crest_elevation_m],
        ])
        add_fault_plane(
            fig,
            polyline=fault_polyline,
            dip_deg=75.0,
            color="#E53935",
            name="FAULT_JWA_01",
        )
