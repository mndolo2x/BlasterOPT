"""
Server-side 3D scene rendering using PyVista and stpyvista.
"""

import streamlit as st
import numpy as np
from src.render3d.bench_model import BenchGeometry
from src.render3d.hole_pattern import HolePattern

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    from stpyvista import stpyvista
    HAS_STPYVISTA = True
except ImportError:
    HAS_STPYVISTA = False


def render_bench_scene(bench: BenchGeometry, pattern: HolePattern) -> None:
    """
    Render 3D bench scene using PyVista and stpyvista for server-side rendering.

    Args:
        bench: BenchGeometry
        pattern: HolePattern
    """
    if not HAS_PYVISTA or not HAS_STPYVISTA:
        st.error("PyVista or stpyvista is not installed. Falling back to Plotly rendering.")
        return

    pv.start_xvfb()  # Enable virtual frame buffer for headless server rendering

    plotter = pv.Plotter(window_size=[700, 580])
    plotter.background_color = "#1E1E1E"

    # 1. Bench Mesh Surface
    subdrill = getattr(pattern, "subdrilling_m", 1.5)
    bottom_z = bench.toe_elevation_m - subdrill

    bench_box = pv.Cube(
        center=(
            bench.length_m / 2.0,
            bench.width_m / 2.0,
            (bench.crest_elevation_m + bottom_z) / 2.0,
        ),
        x_length=bench.length_m,
        y_length=bench.width_m,
        z_length=bench.crest_elevation_m - bottom_z,
    )
    plotter.add_mesh(
        bench_box,
        color="#8B7355",
        opacity=0.65,
        show_edges=True,
        edge_color="#4A3B32",
    )

    # 2. Blastholes as Cylinders Color-coded by Charge Weight
    if pattern.holes:
        charges = [h.charge_kg for h in pattern.holes]
        for h in pattern.holes:
            center = (h.x_m, h.y_m, (h.z_collar_m + h.z_toe_m) / 2.0)
            cylinder = pv.Cylinder(
                center=center,
                direction=(0, 0, -1),
                radius=0.2,
                height=h.depth_m,
            )
            plotter.add_mesh(
                cylinder,
                scalars=[h.charge_kg] * cylinder.n_cells,
                cmap="plasma",
                clim=[min(charges), max(charges) if max(charges) > min(charges) else min(charges) + 1.0],
            )

    plotter.view_isometric()
    plotter.add_scalar_bar(
        title="Charge Weight (kg)",
        position_x=0.2,
        position_y=0.02,
        width=0.6,
        height=0.08,
        title_font_size=12,
        label_font_size=10,
    )

    stpyvista(plotter, key="bench_viewer")
