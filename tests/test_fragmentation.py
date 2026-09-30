"""
Unit tests for src/fragmentation/ package modules.
"""

import pytest
import numpy as np
import pandas as pd
from src.fragmentation import (
    predict_kuz_ram,
    swebrec_cumulative_passing,
    predict_kco,
    compare_distributions,
    parse_wipfrag_data,
    calibrate_rock_factor,
    FragmentationRenderer,
)


def test_kuz_ram():
    res = predict_kuz_ram(powder_factor_kg_m3=0.65, charge_mass_per_hole_kg=150.0, rock_factor_a=8.0)
    assert "d50_mm" in res
    assert res["d50_mm"] > 0.0


def test_swebrec():
    p = swebrec_cumulative_passing(x_size_mm=250.0, x_max_mm=1000.0, x_50_mm=250.0, b_curve_factor=1.0)
    assert abs(p - 50.0) < 0.1


def test_kco():
    res = predict_kco(powder_factor_kg_m3=0.65, charge_mass_per_hole_kg=150.0, rock_factor_a=8.0)
    assert "d50_mm" in res
    assert "d80_mm" in res
    assert "fines_pct_minus_10mm" in res


def test_distributions():
    dists = compare_distributions(x_50=220.0, n_uniformity=1.25, b_swebrec=1.2, x_max=1000.0)
    assert "rosin_rammler" in dists
    assert "swebrec" in dists
    assert "lognormal" in dists


def test_image_analysis():
    metrics = {
        "image_id": "IMG_01",
        "detected_particles_count": 10,
        "particle_areas_px2": [100, 400, 900, 1600, 2500],
        "scale_px_per_cm": 10.0,
    }
    res = parse_wipfrag_data(metrics)
    assert "d50_mm" in res
    assert "d80_mm" in res


def test_calibration():
    cal = calibrate_rock_factor(measured_d50_mm=200.0, powder_factor_kg_m3=0.65, charge_mass_per_hole_kg=150.0)
    assert "calibrated_rock_factor_a" in cal
    assert cal["calibrated_rock_factor_a"] > 0.0


def test_renderer():
    renderer = FragmentationRenderer()
    fig = renderer.plot_distribution_comparison(d50_mm=220.0)
    assert fig is not None
