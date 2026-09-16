"""
Unit tests for Real-Time Adaptive Blast Designer & Risk Controller module.
"""

import os
import sqlite3
import pytest
from src.realtime_adaptive import (
    adjust_charging_plan,
    risk_controller,
    audit_log,
)


def test_adjust_charging_plan_low_rop_increases_powder_factor():
    """Test adjust_charging_plan increases powder factor when penetration rate is low (< 25 m/hr)."""
    current_design = {
        "powder_factor_kg_m3": 0.60,
        "stemming_m": 5.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
    }
    mwd_low_rop = {
        "hole_id": "BH_101",
        "depth": 15.0,
        "penetration_rate": 18.0,  # Low ROP -> Hard rock
        "torque": 1200.0,
        "vibration": 2.5,
    }

    adjusted = adjust_charging_plan(mwd_low_rop, current_design, pf_increase_pct=15.0)

    # 0.60 * (1 + 0.15) = 0.69
    assert adjusted["powder_factor_kg_m3"] > current_design["powder_factor_kg_m3"]
    assert adjusted["powder_factor_kg_m3"] == 0.69
    assert adjusted["adjusted_for_hole"] == "BH_101"


def test_adjust_charging_plan_high_torque_increases_stemming():
    """Test adjust_charging_plan increases stemming length when torque is high (> 1800 N.m)."""
    current_design = {
        "powder_factor_kg_m3": 0.65,
        "stemming_m": 4.5,
    }
    mwd_high_torque = {
        "hole_id": "BH_102",
        "penetration_rate": 35.0,
        "torque": 2200.0,  # High torque -> Fractured ground
        "vibration": 3.0,
    }

    adjusted = adjust_charging_plan(mwd_high_torque, current_design, stemming_increase_m=0.5)

    assert adjusted["stemming_m"] == 5.0
    assert adjusted["powder_factor_kg_m3"] == current_design["powder_factor_kg_m3"]


def test_risk_controller_returns_is_safe_false_when_limits_exceeded():
    """Test risk_controller returns is_safe=False and lists violations when limits are exceeded."""
    unsafe_design = {
        "predicted_ppv_mms": 14.5,        # Exceeds 10.0 limit
        "predicted_airblast_dbl": 126.0,  # Exceeds 120.0 limit
        "predicted_flyrock_m": 290.0,     # Exceeds 250.0 limit
    }
    limits = {
        "max_ppv_mms": 10.0,
        "max_airblast_dbl": 120.0,
        "max_flyrock_m": 250.0,
    }

    res = risk_controller(unsafe_design, regulatory_limits=limits)

    assert isinstance(res, dict)
    assert res["is_safe"] is False
    assert len(res["violations"]) == 3
    assert len(res["recommendations"]) == 3
    assert any("Ground Vibration" in v for v in res["violations"])


def test_risk_controller_safe_design():
    """Test risk_controller returns is_safe=True for safe design."""
    safe_design = {
        "predicted_ppv_mms": 7.5,
        "predicted_airblast_dbl": 114.0,
        "predicted_flyrock_m": 120.0,
    }

    res = risk_controller(safe_design)

    assert res["is_safe"] is True
    assert len(res["violations"]) == 0


def test_audit_log_inserts_row_into_sqlite_database(tmp_path):
    """Test audit_log immutably inserts an audit row into local SQLite database."""
    db_file = str(tmp_path / "test_audit.db")

    entry = audit_log(
        action="ADJUST_CHARGING_PLAN",
        user_id="BLASTER_JWA_01",
        original_value={"powder_factor_kg_m3": 0.60},
        new_value={"powder_factor_kg_m3": 0.69},
        reason_code="MWD_HARD_ROCK_ROP_LOW",
        db_path=db_file,
    )

    assert isinstance(entry, dict)
    assert entry["id"] == 1
    assert entry["action"] == "ADJUST_CHARGING_PLAN"
    assert entry["user_id"] == "BLASTER_JWA_01"
    assert entry["reason_code"] == "MWD_HARD_ROCK_ROP_LOW"

    # Query SQLite database file directly
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id, action, reason_code FROM audit_log WHERE id = 1")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 1
    assert row[1] == "BLASTER_JWA_01"
    assert row[2] == "ADJUST_CHARGING_PLAN"
    assert row[3] == "MWD_HARD_ROCK_ROP_LOW"
