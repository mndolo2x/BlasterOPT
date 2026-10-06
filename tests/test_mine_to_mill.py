"""
Unit test suite for Mine-to-Mill Cost Model (src/mine_to_mill.py).
"""

import pytest
from src.mine_to_mill import milling_cost, total_cost_per_tonne


def test_milling_cost_concave_growth():
    """Test milling_cost exhibits concave growth (power 0.8) and returns expected cost."""
    cost_200 = milling_cost(d80_cm=20.0)  # D80 = 200 mm
    cost_400 = milling_cost(d80_cm=40.0)  # D80 = 400 mm

    assert cost_200 > 2.50
    assert cost_400 > cost_200
    # Concave growth check: coarse cost ratio should be < 2.0x for 2.0x D80 ratio
    assert (cost_400 / cost_200) < 2.0


def test_total_cost_per_tonne_in_target_range():
    """Test total_cost_per_tonne returns total unit cost in $4.00 - $7.00/t target range."""
    inp = {
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "bench_height_m": 15.0,
        "hole_diameter_mm": 250.0,
        "d80_mm": 250.0,
    }

    res = total_cost_per_tonne(inp)
    assert 4.00 <= res["total_cost_usd_t"] <= 7.00
    assert 0.20 <= res["drilling_cost_usd_t"] <= 0.50
    assert 0.20 <= res["explosive_cost_usd_t"] <= 0.50
    assert 0.40 <= res["digging_cost_usd_t"] <= 0.70
    assert 0.60 <= res["hauling_cost_usd_t"] <= 1.00
    assert 0.40 <= res["crushing_cost_usd_t"] <= 0.80
    assert 2.00 <= res["milling_cost_usd_t"] <= 4.00
