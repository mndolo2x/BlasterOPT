"""
Unit tests for 3D render visuals (src/render3d/renderer.py & bench_model.py).
"""

from src.render3d.bench_model import BenchGeometry, build_bench_mesh
from src.render3d.hole_pattern import HolePattern, generate_holes
from src.render3d.renderer import render_bench_and_holes


def test_bench_mesh_subdrill_bottom_elevation():
    """Verify bench mesh extends down to toe_elevation_m minus subdrill_m."""
    bench = BenchGeometry(
        bench_id="B14",
        crest_elevation_m=15.0,
        toe_elevation_m=0.0,
    )
    mesh = build_bench_mesh(bench, subdrill_m=1.5)

    assert "bottom_z" in mesh
    assert mesh["bottom_z"] == -1.5


def test_render_bench_and_holes_figure_layout():
    """Verify figure creation with 14pt axis titles and 12pt tick fonts."""
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

    scene = fig.layout.scene
    assert scene.xaxis.title.font.size == 14
    assert scene.xaxis.tickfont.size == 12
    assert len(fig.data) > 0
