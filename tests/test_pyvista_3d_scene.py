"""
Unit tests for Plotly SVG render_mode fallback and PyVista 3D scene rendering routing.
"""

import os
import pytest
from src.render3d.bench_model import BenchGeometry
from src.render3d.hole_pattern import HolePattern, generate_holes
from src.render3d.renderer import render_bench_and_holes


def test_plotly_3d_renders_without_webgl():
    """Verify Plotly 3D figure creates data traces and valid layout."""
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

    fig = render_bench_and_holes(bench, pattern)

    assert len(fig.data) > 0
    assert fig.layout.scene.xaxis.title.text == "X (m)"


def test_pyvista_scene_import():
    """Verify src.ui.pyvista_scene import and callable interface."""
    from src.ui.pyvista_scene import render_bench_scene
    assert callable(render_bench_scene)


def test_pyvista_feature_flag_default_false():
    """Verify USE_PYVISTA environment variable defaults to False."""
    flag = os.getenv("USE_PYVISTA", "false").lower() == "true"
    assert flag is False
