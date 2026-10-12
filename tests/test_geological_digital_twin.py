"""
Unit tests for 3D geological digital twin overlays and irregular bench terrain mesh.
"""

import numpy as np
import plotly.graph_objects as go
from src.render3d.bench_model import BenchGeometry, build_bench_mesh
from src.render3d.geology_overlay import add_rock_layer, add_fault_plane, add_geological_overlays
from src.render3d.hole_pattern import HolePattern, generate_holes
from src.render3d.renderer import render_bench_and_holes


def test_topo_mesh_and_collar_z_lookup():
    """Verify irregular terrain mesh generation and collar Z coordinate lookup."""
    bench = BenchGeometry(
        bench_id="B14",
        crest_elevation_m=15.0,
        toe_elevation_m=0.0,
    )
    mesh = build_bench_mesh(bench, subdrill_m=1.5)

    assert "get_collar_z" in mesh
    z_collar = mesh["get_collar_z"](10.0, 5.0)
    assert isinstance(z_collar, float)
    assert abs(z_collar - 15.0) < 1.0


def test_add_geological_overlays_adds_traces():
    """Verify add_geological_overlays adds 3D rock layer meshes and fault plane to Plotly figure."""
    fig = go.Figure()
    add_geological_overlays(
        fig=fig,
        bench_length_m=50.0,
        bench_width_m=30.0,
        crest_elevation_m=15.0,
        toe_elevation_m=0.0,
        show_geology=True,
        show_fault=True,
    )

    # 3 rock layers (Kimberlite, Dolomite, Granite) + 1 fault plane FAULT_JWA_01 = 4 traces
    assert len(fig.data) == 4
    trace_names = [t.name for t in fig.data]
    assert "Kimberlite Ore" in trace_names
    assert "Dolomite Waste" in trace_names
    assert "Granite Contact" in trace_names
    assert "FAULT_JWA_01" in trace_names


def test_render_bench_and_holes_with_geology_toggles():
    """Verify render_bench_and_holes accepts geology and fault toggles."""
    bench = BenchGeometry(
        bench_id="B14",
        crest_elevation_m=15.0,
        toe_elevation_m=0.0,
    )
    pattern = HolePattern(
        pattern_type="staggered",
        burden_m=4.2,
        spacing_m=5.1,
        num_rows=8,
        holes_per_row=6,
        stemming_m=3.0,
        hole_depth_m=16.5,
        hole_angle_deg=90.0,
        powder_factor_kg_m3=0.65,
        bench_height_m=15.0,
        subdrilling_m=1.5,
    )
    pattern = generate_holes(pattern)

    fig_full = render_bench_and_holes(bench, pattern, show_geology=True, show_fault=True)
    fig_no_geo = render_bench_and_holes(bench, pattern, show_geology=False, show_fault=False)

    assert len(fig_full.data) > len(fig_no_geo.data)
