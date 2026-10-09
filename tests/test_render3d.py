"""
Unit tests for 3D blast rendering module (src/render3d/).
"""
import numpy as np
import pytest
from src.render3d.bench_model import BenchGeometry, compute_bench_height, compute_face_offset
from src.render3d.hole_pattern import HolePattern, generate_holes, pattern_summary
from src.render3d.volume import compute_blast_volume, compute_blast_tonnage


def test_bench_height_positive():
    bench = BenchGeometry(bench_id="B14", crest_elevation_m=15.0, toe_elevation_m=0.0)
    assert compute_bench_height(bench) == 15.0


def test_bench_height_negative_raises():
    bench = BenchGeometry(bench_id="B14", crest_elevation_m=0.0, toe_elevation_m=15.0)
    with pytest.raises(ValueError):
        compute_bench_height(bench)


def test_face_offset_formula():
    bench = BenchGeometry(bench_id="B14", crest_elevation_m=15.0, toe_elevation_m=0.0, face_angle_deg=75.0)
    offset = compute_face_offset(bench)
    expected = 15.0 / np.tan(np.radians(75.0))
    assert abs(offset - expected) < 1e-6


def test_hole_pattern_generates_correct_count():
    pattern = HolePattern(
        pattern_type="staggered", burden_m=4.2, spacing_m=5.1,
        num_rows=8, holes_per_row=6, stemming_m=3.0,
        hole_depth_m=16.5, hole_angle_deg=90.0, powder_factor_kg_m3=0.65,
        bench_height_m=15.0, subdrilling_m=1.5,
    )
    pattern = generate_holes(pattern)
    assert len(pattern.holes) == 48


def test_spacing_greater_than_burden():
    pattern = HolePattern(
        pattern_type="square", burden_m=4.2, spacing_m=5.1,
        num_rows=4, holes_per_row=4, stemming_m=3.0,
        hole_depth_m=16.5, hole_angle_deg=90.0, powder_factor_kg_m3=0.65,
        bench_height_m=15.0, subdrilling_m=1.5,
    )
    pattern = generate_holes(pattern)
    assert pattern.spacing_m >= pattern.burden_m


def test_volume_calculation():
    pattern = HolePattern(
        pattern_type="square", burden_m=4.0, spacing_m=5.0,
        num_rows=4, holes_per_row=4, stemming_m=3.0,
        hole_depth_m=16.5, hole_angle_deg=90.0, powder_factor_kg_m3=0.65,
        bench_height_m=15.0, subdrilling_m=1.5,
    )
    pattern = generate_holes(pattern)
    expected = 4.0 * 5.0 * 15.0 * 16
    assert abs(compute_blast_volume(pattern) - expected) < 1e-6
    assert compute_blast_tonnage(pattern) > 0


def test_pattern_summary_keys():
    pattern = HolePattern(
        pattern_type="square", burden_m=4.0, spacing_m=5.0,
        num_rows=4, holes_per_row=4, stemming_m=3.0,
        hole_depth_m=16.5, hole_angle_deg=90.0, powder_factor_kg_m3=0.65,
        bench_height_m=15.0, subdrilling_m=1.5,
    )
    pattern = generate_holes(pattern)
    summary = pattern_summary(pattern)
    assert "num_holes" in summary
    assert "total_charge_kg" in summary
    assert "max_delay_ms" in summary
