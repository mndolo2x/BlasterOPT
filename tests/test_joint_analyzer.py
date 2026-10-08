"""
Unit tests for Joint Set Analyzer calculation logic and block volume estimation.
"""

import pytest
import pandas as pd


def analyze_joint_sets_df(df: pd.DataFrame) -> dict:
    """Computes mean orientation, spacing, persistence, and estimated block volume."""
    if len(df) == 0:
        return {}

    block_vol = float(df["spacing_m"].nsmallest(3).prod()) if len(df) >= 3 else None

    size_class = None
    if block_vol is not None:
        if block_vol < 0.01:
            size_class = "Very Small"
        elif block_vol < 0.1:
            size_class = "Small"
        elif block_vol < 1.0:
            size_class = "Medium"
        elif block_vol < 10.0:
            size_class = "Large"
        else:
            size_class = "Very Large"

    return {
        "num_sets": len(df),
        "mean_dip": float(df["dip_deg"].mean()),
        "mean_dip_direction": float(df["dip_direction_deg"].mean()),
        "mean_spacing_m": float(df["spacing_m"].mean()),
        "mean_persistence_m": float(df["persistence_m"].mean()),
        "block_volume_m3": block_vol,
        "block_size_class": size_class,
    }


def test_joint_analyzer_two_sets():
    """Test analysis for initial 2 default joint sets."""
    df2 = pd.DataFrame([
        {"set_id": 1, "dip_deg": 60, "dip_direction_deg": 120, "spacing_m": 1.0, "persistence_m": 5.0},
        {"set_id": 2, "dip_deg": 45, "dip_direction_deg": 240, "spacing_m": 0.8, "persistence_m": 6.0},
    ])

    res = analyze_joint_sets_df(df2)
    assert res["num_sets"] == 2
    assert res["mean_dip"] == 52.5
    assert res["mean_dip_direction"] == 180.0
    assert res["block_volume_m3"] is None


def test_joint_analyzer_three_sets_block_volume():
    """Test analysis for 3 joint sets with block volume calculation."""
    df3 = pd.DataFrame([
        {"set_id": 1, "dip_deg": 60, "dip_direction_deg": 120, "spacing_m": 1.0, "persistence_m": 5.0},
        {"set_id": 2, "dip_deg": 45, "dip_direction_deg": 240, "spacing_m": 0.8, "persistence_m": 6.0},
        {"set_id": 3, "dip_deg": 75, "dip_direction_deg": 300, "spacing_m": 1.5, "persistence_m": 4.0},
    ])

    res = analyze_joint_sets_df(df3)
    assert res["num_sets"] == 3
    # product of smallest 3 spacings: 1.0 * 0.8 * 1.5 = 1.2 m3
    assert abs(res["block_volume_m3"] - 1.2) < 1e-4
    assert res["block_size_class"] == "Large"


def test_block_volume_updates_when_spacing_changes():
    """Test that block volume and size class update when spacing changes."""
    df3_small = pd.DataFrame([
        {"set_id": 1, "dip_deg": 60, "dip_direction_deg": 120, "spacing_m": 0.2, "persistence_m": 2.0},
        {"set_id": 2, "dip_deg": 45, "dip_direction_deg": 240, "spacing_m": 0.3, "persistence_m": 2.0},
        {"set_id": 3, "dip_deg": 75, "dip_direction_deg": 300, "spacing_m": 0.4, "persistence_m": 2.0},
    ])

    res = analyze_joint_sets_df(df3_small)
    # product: 0.2 * 0.3 * 0.4 = 0.024 m3
    assert abs(res["block_volume_m3"] - 0.024) < 1e-4
    assert res["block_size_class"] == "Small"
