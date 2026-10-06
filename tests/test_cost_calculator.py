"""
Unit test suite for Cost Calculator Sanity Checks (src/economics/cost_calculator.py).
"""

import os
import pytest
from src.economics.cost_calculator import calculate_cost, MIN_REASONABLE_COST, MAX_REASONABLE_COST, CostEstimate


def test_cost_sanity_check_rejects_negative():
    """Test cost calculator rejects negative cost predictions as INVALID."""
    design = {"cost_per_tonne_usd": -5.00}
    res = calculate_cost(design)
    assert res.status == "INVALID"
    assert res.cost_per_tonne_usd == -5.00
    assert any("outside the reasonable range" in w for w in res.warnings)


def test_cost_sanity_check_rejects_zero():
    """Test cost calculator rejects $0 cost predictions as INVALID."""
    design = {"cost_per_tonne_usd": 0.00}
    res = calculate_cost(design)
    assert res.status == "INVALID"


def test_cost_sanity_check_rejects_huge_values():
    """Test cost calculator rejects absurdly high cost predictions (e.g., $665/t) as INVALID."""
    design = {"cost_per_tonne_usd": 665.34}
    res = calculate_cost(design)
    assert res.status == "INVALID"


def test_cost_sanity_check_accepts_reasonable():
    """Test cost calculator accepts reasonable costs ($1.00 - $10.00 / t) as OK."""
    design = {"cost_per_tonne_usd": 4.80}
    res = calculate_cost(design)
    assert res.status == "OK"
    assert res.cost_per_tonne_usd == 4.80
    assert res.warnings == []


def test_cost_increases_with_powder_factor():
    """Test that drill-and-blast cost estimate increases monotonically with powder factor."""
    design_low_pf = {
        "powder_factor_kg_m3": 0.50,
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "bench_height_m": 15.0,
        "hole_diameter_mm": 250.0,
    }
    design_high_pf = {
        "powder_factor_kg_m3": 0.80,
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "bench_height_m": 15.0,
        "hole_diameter_mm": 250.0,
    }

    res_low = calculate_cost(design_low_pf)
    res_high = calculate_cost(design_high_pf)

    assert res_high.cost_per_tonne_usd > res_low.cost_per_tonne_usd


def test_audit_log_written_for_out_of_range_cost():
    """Test out-of-range prediction generates an audit event."""
    design = {"cost_per_tonne_usd": 850.00, "burden_m": 3.0}
    res = calculate_cost(design, site_id="Jwaneng Mine")
    assert res.status == "INVALID"
