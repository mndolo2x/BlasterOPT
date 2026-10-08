"""
Functional unit tests for RMR calculation button logic and rating functions.
"""

import pytest
from src.geology import (
    rate_ucs,
    rate_rqd,
    rate_spacing,
    rate_condition,
    rate_groundwater,
    rate_orientation,
)


def test_rmr_changes_with_inputs():
    """
    Compute RMR with two different input sets.
    The results must be different.
    """
    # Input set A
    rmr_a = (
        rate_ucs(120) + rate_rqd(75) + rate_spacing(0.5)
        + rate_condition("1-3 m", "0.1-1.0 mm", "Rough", "None", "Slightly weathered")
        + rate_groundwater("Damp") + rate_orientation("Fair")
    )

    # Input set B — different UCS, RQD, spacing, groundwater
    rmr_b = (
        rate_ucs(30) + rate_rqd(40) + rate_spacing(0.1)
        + rate_condition("10-20 m", "1-5 mm", "Smooth", "Soft filling < 5 mm", "Highly weathered")
        + rate_groundwater("Flowing") + rate_orientation("Very unfavorable")
    )

    assert rmr_a != rmr_b, (
        f"RMR is identical for two different input sets ({rmr_a} vs {rmr_b}). "
        f"The calculation is not using the inputs."
    )
    assert rmr_a > rmr_b, "Better rock should have a higher RMR"

    print(f"Input set A (good rock): RMR = {rmr_a}")
    print(f"Input set B (poor rock): RMR = {rmr_b}")
    print("✅ RMR responds to input changes")


def test_rmr_groundwater_changes_score():
    """Changing only groundwater must change the RMR."""
    assert rate_groundwater("Completely dry") == 15
    assert rate_groundwater("Damp") == 10
    assert rate_groundwater("Wet") == 7
    assert rate_groundwater("Dripping") == 4
    assert rate_groundwater("Flowing") == 0


def test_rmr_orientation_is_negative():
    """Orientation ratings must all be zero or negative."""
    for orientation in ["Very favorable", "Favorable", "Fair", "Unfavorable", "Very unfavorable"]:
        score = rate_orientation(orientation)
        assert score <= 0, f"{orientation} returned {score}, must be ≤ 0"
