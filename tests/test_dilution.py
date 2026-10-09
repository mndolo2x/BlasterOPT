"""
Unit tests for ore dilution estimation model in src/geology.py.
"""

import pytest
from src.geology import estimate_dilution


def test_dilution_changes_with_rmr():
    """Poorer rock must produce more dilution."""
    good = estimate_dilution(
        powder_factor=0.65, rmr=80, ore_width_m=15.0,
        boundary_type="Sharp contact", stemming_ratio=0.8,
    )
    poor = estimate_dilution(
        powder_factor=0.65, rmr=30, ore_width_m=15.0,
        boundary_type="Sharp contact", stemming_ratio=0.8,
    )
    assert poor["dilution_pct"] > good["dilution_pct"], (
        f"Poor rock diluted less ({poor['dilution_pct']}) "
        f"than good rock ({good['dilution_pct']})"
    )


def test_dilution_changes_with_boundary():
    """Faulted contacts must dilute more than sharp contacts."""
    sharp = estimate_dilution(
        powder_factor=0.65, rmr=62, ore_width_m=15.0,
        boundary_type="Sharp contact", stemming_ratio=0.8,
    )
    faulted = estimate_dilution(
        powder_factor=0.65, rmr=62, ore_width_m=15.0,
        boundary_type="Faulted contact", stemming_ratio=0.8,
    )
    assert faulted["dilution_pct"] > sharp["dilution_pct"]


def test_dilution_changes_with_powder_factor():
    """Higher powder factor must increase dilution."""
    low = estimate_dilution(
        powder_factor=0.40, rmr=62, ore_width_m=15.0,
        boundary_type="Sharp contact", stemming_ratio=0.8,
    )
    high = estimate_dilution(
        powder_factor=1.20, rmr=62, ore_width_m=15.0,
        boundary_type="Sharp contact", stemming_ratio=0.8,
    )
    assert high["dilution_pct"] > low["dilution_pct"]


def test_dilution_within_bounds():
    """Dilution must always be between 2% and 25%."""
    # Best case
    best = estimate_dilution(
        powder_factor=0.20, rmr=100, ore_width_m=50.0,
        boundary_type="Sharp contact", stemming_ratio=1.5,
    )
    assert best["dilution_pct"] >= 2.0

    # Worst case
    worst = estimate_dilution(
        powder_factor=1.50, rmr=0, ore_width_m=1.0,
        boundary_type="Faulted contact", stemming_ratio=0.3,
    )
    assert worst["dilution_pct"] <= 25.0
