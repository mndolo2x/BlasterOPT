"""
Unit tests for the Direct-to-Drill Connectivity module.
"""

import pytest
from datetime import datetime
from src.drill_connectivity import connect_to_sandvik, connect_to_epiroc, sync_design_to_drill


def test_connect_to_sandvik_returns_fleet_and_iso_flag():
    """Test connect_to_sandvik returns fleet list and ISO 15143 compliance flag."""
    res = connect_to_sandvik()

    assert isinstance(res, dict)
    assert res["vendor"] == "Sandvik"
    assert "status" in res
    assert "fleet" in res
    assert isinstance(res["fleet"], list)
    assert len(res["fleet"]) > 0
    assert res["iso_15143_compliant"] is True


def test_connect_to_epiroc_returns_fleet():
    """Test connect_to_epiroc returns active Epiroc fleet list."""
    res = connect_to_epiroc()

    assert isinstance(res, dict)
    assert res["vendor"] == "Epiroc"
    assert "status" in res
    assert "fleet" in res
    assert isinstance(res["fleet"], list)
    assert len(res["fleet"]) > 0


def test_sync_design_to_drill_handles_gracefully():
    """Test sync_design_to_drill handles design push requests and credentials gracefully."""
    design_payload = {"design_id": "PATTERN_TEST_01", "num_holes": 32}

    res_sandvik = sync_design_to_drill(design_payload, drill_id="SANDVIK_DR412i_01", vendor="sandvik")
    assert isinstance(res_sandvik, dict)
    assert res_sandvik["status"] in ["success", "synced_offline"]
    assert res_sandvik["drill_id"] == "SANDVIK_DR412i_01"
    assert res_sandvik["holes_synced"] == 32
    assert "sync_timestamp" in res_sandvik
    # Verify sync_timestamp is a valid ISO timestamp
    parsed_ts = datetime.fromisoformat(res_sandvik["sync_timestamp"])
    assert parsed_ts is not None

    res_epiroc = sync_design_to_drill(design_payload, drill_id="EPIROC_PV271_01", vendor="epiroc")
    assert isinstance(res_epiroc, dict)
    assert res_epiroc["status"] in ["success", "synced_offline"]
    assert res_epiroc["drill_id"] == "EPIROC_PV271_01"


def test_sync_timestamp_is_valid_iso8601():
    """Test that sync_design_to_drill returns a valid ISO 8601 timestamp string for sync_timestamp."""
    res = sync_design_to_drill({"design_id": "ISO_TEST_PATTERN", "num_holes": 10}, drill_id="SANDVIK_DR412i_01")

    assert "sync_timestamp" in res
    timestamp_str = res["sync_timestamp"]
    assert isinstance(timestamp_str, str)
    assert not timestamp_str.startswith("python-requests")

    parsed_dt = datetime.fromisoformat(timestamp_str)
    assert isinstance(parsed_dt, datetime)


def test_sync_design_to_drill_invalid_vendor_fallback():
    """Test sync_design_to_drill falls back gracefully when given unknown vendor."""
    res = sync_design_to_drill({"num_holes": 16}, drill_id="RIG_UNKNOWN", vendor="unknown_vendor")

    assert isinstance(res, dict)
    assert res["status"] in ["success", "synced_offline"]
    assert res["holes_synced"] == 16
