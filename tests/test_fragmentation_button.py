"""
Functional unit tests for Advanced Fragmentation Kuz-Ram button calculation logic.
"""

import pytest
from src.physics_core import kuznetsov_x50, cunningham_uniformity, rosin_rammler_d80


def calculate_fragmentation_d50(powder_factor_kg_m3: float) -> float:
    """Computes Kuz-Ram d50 mm for a given powder factor."""
    rock_factor = 8.0
    burden = 4.2
    spacing = 5.1
    hole_depth = 16.5
    explosive_rws = 115.0

    bench_h = 15.0
    hole_vol = burden * spacing * bench_h
    charge_mass = powder_factor_kg_m3 * hole_vol

    x50_cm = kuznetsov_x50(
        rock_factor_a=rock_factor,
        burden_m=burden,
        spacing_m=spacing,
        hole_depth_m=hole_depth,
        charge_mass_kg=charge_mass,
        explosive_rws=explosive_rws,
    )
    return round(x50_cm * 10.0, 1)


def test_fragmentation_changes_with_powder_factor():
    """Higher powder factor must produce finer fragmentation (smaller d50)."""
    d1 = calculate_fragmentation_d50(0.65)
    d2 = calculate_fragmentation_d50(0.80)

    assert d2 < d1, f"Expected D2 ({d2} mm) < D1 ({d1} mm) for higher powder factor"


def test_kuz_ram_uniformity_index():
    """Uniformity index n must be positive."""
    n = cunningham_uniformity(
        burden_m=4.2,
        spacing_m=5.1,
        hole_diameter_mm=165.0,
        bench_height_m=15.0,
        charge_length_m=13.5,
    )
    assert n > 0.5
