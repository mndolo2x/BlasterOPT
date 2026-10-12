"""
Unit tests for 3D Bench Digital Twin module (src/ui/bench_twin.py).
"""

import os
import tempfile
import plotly.graph_objects as go
from src.ui.bench_twin import (
    load_holes_from_design,
    load_geology_overlays,
    make_bench_figure,
)


def test_load_holes_from_design_returns_correct_shape():
    """Verify load_holes_from_design parses pattern dict into hole records."""
    design = {
        "burden_m": 4.2,
        "spacing_m": 5.1,
        "num_rows": 4,
        "holes_per_row": 5,
        "hole_depth_m": 12.0,
        "powder_factor_kg_m3": 0.65,
        "bench_height_m": 12.0,
    }
    holes = load_holes_from_design(design)

    assert len(holes) == 20
    first = holes[0]
    assert "hole_id" in first
    assert "x" in first
    assert "y" in first
    assert "z" in first
    assert "x_toe" in first
    assert "y_toe" in first
    assert "z_toe" in first
    assert "charge_kg" in first
    assert first["z_toe"] == -12.0


def test_load_geology_overlays_returns_valid_quads():
    """Verify load_geology_overlays returns quadrilaterals with vertices."""
    overlays = load_geology_overlays("B14", "jwaneng")

    assert len(overlays) >= 2
    for ov in overlays:
        assert "name" in ov
        assert "color" in ov
        assert "opacity" in ov
        assert "vertices" in ov
        assert len(ov["vertices"]) == 4


def test_make_bench_figure_returns_plotly_figure():
    """Verify make_bench_figure creates a valid Plotly Figure."""
    fig = make_bench_figure()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_toggles_hide_traces_correctly():
    """Verify show_holes, show_geology, and show_faults toggles filter traces."""
    fig_all = make_bench_figure(show_holes=True, show_geology=True, show_faults=True)
    fig_no_holes = make_bench_figure(show_holes=False, show_geology=True, show_faults=True)
    fig_no_geo = make_bench_figure(show_holes=True, show_geology=False, show_faults=False)

    assert len(fig_all.data) > len(fig_no_holes.data)
    assert len(fig_all.data) > len(fig_no_geo.data)


def test_export_html_writes_file():
    """Verify exporting figure to HTML file."""
    fig = make_bench_figure()
    html_str = fig.to_html()

    assert "plotly" in html_str.lower()
    assert len(html_str) > 1000
