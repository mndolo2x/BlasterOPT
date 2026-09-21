"""
Unit tests for Electronic Detonator Integration module.
"""

import pytest
from src.detonator_integration import (
    validate_sequence,
    upload_timing_sequence,
    download_firing_confirmation,
)


def test_validate_sequence_compliant():
    """Test validate_sequence returns True and empty violations list for valid timing sequence."""
    compliant_seq = {
        "hole_delay_ms": 17.0,
        "row_delay_ms": 42.0,
        "max_charge_per_delay_kg": 640.0,
        "predicted_ppv_mms": 8.2,
        "predicted_airblast_dbl": 112.5,
    }

    is_valid, violations = validate_sequence(compliant_seq)

    assert is_valid is True
    assert isinstance(violations, list)
    assert len(violations) == 0


def test_validate_sequence_non_compliant_violations():
    """Test validate_sequence returns False and populates specific violation messages when thresholds are exceeded."""
    violating_seq = {
        "hole_delay_ms": 4.0,  # Below 8 ms limit
        "row_delay_ms": 15.0,  # Below 25 ms limit
        "predicted_ppv_mms": 14.5,  # Exceeds 10.0 mm/s limit
        "predicted_airblast_dbl": 125.0,  # Exceeds 120 dBL limit
    }

    is_valid, violations = validate_sequence(violating_seq)

    assert is_valid is False
    assert isinstance(violations, list)
    assert len(violations) == 4
    assert any("Inter-hole delay" in v for v in violations)
    assert any("Inter-row delay" in v for v in violations)
    assert any("ground vibration" in v for v in violations)
    assert any("airblast" in v for v in violations)


def test_upload_timing_sequence_vendors():
    """Test upload_timing_sequence works for AEL, BME, and Orica systems without raising exceptions."""
    seq_payload = {"hole_delay_ms": 25, "row_delay_ms": 42}

    res_ael = upload_timing_sequence("AEL IntelliShot", seq_payload, blast_id="BLAST_01")
    assert isinstance(res_ael, dict)
    assert "AEL" in res_ael["detonator_system"]
    assert res_ael["blast_id"] == "BLAST_01"

    res_bme = upload_timing_sequence("BME AXXIS", seq_payload, blast_id="BLAST_02")
    assert isinstance(res_bme, dict)
    assert "BME" in res_bme["detonator_system"]

    res_orica = upload_timing_sequence("Orica i-kon", seq_payload, blast_id="BLAST_03")
    assert isinstance(res_orica, dict)
    assert "Orica" in res_orica["detonator_system"]


def test_upload_blocked_on_invalid_sequence():
    """Test upload_timing_sequence blocks invalid sequence and aborts upload."""
    invalid_seq = {
        "hole_delay_ms": 3.0,  # Below 8 ms limit
        "row_delay_ms": 10.0,  # Below 25 ms limit
        "predicted_ppv_mms": 15.0,  # Exceeds limit
    }

    res = upload_timing_sequence("AEL IntelliShot", invalid_seq, blast_id="BLAST_INVALID_01")

    assert isinstance(res, dict)
    assert res["status"] == "blocked"
    assert res["sequence_valid"] is False
    assert len(res["violations"]) >= 3
    assert "rejected" in res["message"] or "aborted" in res["message"]


def test_upload_proceeds_on_valid_sequence():
    """Test upload_timing_sequence proceeds successfully when given a valid sequence."""
    valid_seq = {
        "hole_delay_ms": 17.0,
        "row_delay_ms": 42.0,
        "max_charge_per_delay_kg": 640.0,
        "predicted_ppv_mms": 8.0,
        "predicted_airblast_dbl": 115.0,
    }

    res = upload_timing_sequence("BME AXXIS", valid_seq, blast_id="BLAST_VALID_01")

    assert isinstance(res, dict)
    assert res["status"] != "blocked"
    assert res["sequence_valid"] is True
    assert len(res["violations"]) == 0
    assert "successfully uploaded" in res["message"]


def test_blocked_upload_logs_audit_event(monkeypatch):
    """Test that attempting to upload an invalid sequence logs a timing_sequence_blocked audit event."""
    logged_events = []

    class MockAuditService:
        def log_event(self, event_type, user_id, payload, design_id=None):
            logged_events.append({
                "event_type": event_type,
                "user_id": user_id,
                "payload": payload,
                "design_id": design_id,
            })

    monkeypatch.setattr("src.services.audit_service.AuditService", MockAuditService)

    invalid_seq = {"hole_delay_ms": 2.0}
    upload_timing_sequence("Orica i-kon", invalid_seq, blast_id="BLAST_AUDIT_01")

    assert len(logged_events) == 1
    assert logged_events[0]["event_type"] == "timing_sequence_blocked"
    assert logged_events[0]["payload"]["reason"] == "invalid_sequence"
    assert logged_events[0]["design_id"] == "BLAST_AUDIT_01"


def test_download_firing_confirmation():
    """Test download_firing_confirmation returns structured post-blast diagnostics dictionary."""
    conf = download_firing_confirmation("BME AXXIS", blast_id="BLAST_02")

    assert isinstance(conf, dict)
    assert conf["blast_id"] == "BLAST_02"
    assert conf["firing_status"] == "SUCCESSFUL_INITIATION"
    assert conf["detonators_fired"] == 128
    assert conf["misfires_count"] == 0
    assert "measured_ppv_mms" in conf
