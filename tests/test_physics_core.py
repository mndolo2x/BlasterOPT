"""
Unit tests for src/physics_core.py.
"""
import pytest
from src.physics_core import (
    kuznetsov_x50,
    cunningham_uniformity,
    rosin_rammler_d80,
    usbm_ppv,
    siskind_airblast,
    lundborg_flyrock,
    total_cost_per_tonne,
)


def test_kuznetsov_x50():
    x50 = kuznetsov_x50(
        rock_factor_a=8.0,
        burden_m=6.0,
        spacing_m=7.0,
        hole_depth_m=15.0,
        charge_mass_kg=320.0,
        explosive_rws=100.0,
    )
    assert 10.0 < x50 < 100.0, f"Expected reasonable x50 in cm, got {x50}"


def test_cunningham_uniformity():
    n = cunningham_uniformity(
        burden_m=6.0,
        spacing_m=7.0,
        hole_diameter_mm=250.0,
        bench_height_m=15.0,
        charge_length_m=10.0,
    )
    assert 0.5 <= n <= 2.5, f"Expected n in [0.5, 2.5], got {n}"


def test_rosin_rammler_d80():
    d80 = rosin_rammler_d80(x50_cm=20.0, n=1.2)
    assert d80 > 20.0, f"D80 ({d80}) must be greater than x50 (20.0)"


def test_usbm_ppv():
    ppv = usbm_ppv(max_charge_per_delay_kg=640.0, distance_m=450.0)
    assert 0.1 < ppv < 50.0, f"Expected reasonable PPV in mm/s, got {ppv}"


def test_siskind_airblast():
    airblast = siskind_airblast(max_charge_per_delay_kg=640.0, distance_m=450.0)
    assert 40.0 <= airblast <= 140.0, f"Expected airblast in [40, 140] dB, got {airblast}"


def test_lundborg_flyrock():
    flyrock = lundborg_flyrock(charge_mass_kg=320.0, stemming_m=5.0, burden_m=6.0)
    assert 5.0 <= flyrock <= 500.0, f"Expected flyrock in [5, 500] m, got {flyrock}"


def test_total_cost_per_tonne():
    cost = total_cost_per_tonne(
        drilling_cost_per_m=12.0,
        hole_depth_m=15.0,
        n_holes=50,
        explosive_cost_per_kg=1.5,
        charge_mass_per_hole_kg=320.0,
        labor_cost=500.0,
        tonnage=10000.0,
    )
    assert 0.5 < cost < 50.0, f"Expected reasonable cost per tonne in USD, got {cost}"
