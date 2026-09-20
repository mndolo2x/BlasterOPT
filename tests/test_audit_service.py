"""
Unit tests for immutable hash-chained audit service (src/services/audit_service.py).
"""

import os
import json
import pytest
from datetime import datetime, timezone
from src.services.audit_service import (
    AuditEvent,
    AuditService,
    compute_payload_hash,
    compute_event_hash,
    verify_chain,
)


def test_audit_event_model():
    payload = {"design_id": "PATTERN_001", "burden_m": 6.0}
    p_hash = compute_payload_hash(payload)

    evt = AuditEvent(
        event_type="prediction",
        user_id="ENGINEER_01",
        payload_hash=p_hash,
        payload=payload,
    )

    assert evt.event_id is not None
    assert evt.event_type == "prediction"
    assert evt.payload_hash == p_hash
    assert evt.previous_event_hash is None


def test_audit_chain_verifies(tmp_path):
    audit_dir = str(tmp_path / "audit")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    service = AuditService(audit_dir=audit_dir)

    # Log event 1
    evt1 = service.log_event(
        event_type="prediction",
        user_id="ENGINEER_ALICE",
        payload={"action": "predict", "ppv": 4.5},
        design_id="PATTERN_101",
        design_version=1,
        date_str=today_str,
    )

    assert evt1.previous_event_hash is None

    # Log event 2
    evt2 = service.log_event(
        event_type="approval_requested",
        user_id="ENGINEER_ALICE",
        payload={"action": "submit_approval", "status": "PENDING"},
        design_id="PATTERN_101",
        design_version=1,
        date_str=today_str,
    )

    assert evt2.previous_event_hash is not None
    assert evt2.previous_event_hash == compute_event_hash(evt1.model_dump())

    # Verify chain integrity
    is_valid = verify_chain(date_str=today_str, audit_dir=audit_dir)
    assert is_valid is True


def test_audit_chain_detects_tampering(tmp_path):
    audit_dir = str(tmp_path / "audit")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    service = AuditService(audit_dir=audit_dir)

    service.log_event("prediction", "USER_01", {"score": 10}, date_str=today_str)
    service.log_event("approval", "USER_02", {"score": 20}, date_str=today_str)

    log_filepath = os.path.join(audit_dir, f"{today_str}.jsonl")

    # Read lines and tamper with payload in line 0
    with open(log_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    line_0_dict = json.loads(lines[0])
    line_0_dict["payload"]["score"] = 999  # Tamper payload value without updating payload_hash

    lines[0] = json.dumps(line_0_dict) + "\n"

    with open(log_filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Chain verification MUST fail
    is_valid = verify_chain(date_str=today_str, audit_dir=audit_dir)
    assert is_valid is False


def test_verify_chain_detects_line_deletion_tampering(tmp_path):
    audit_dir = str(tmp_path / "audit")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    service = AuditService(audit_dir=audit_dir)

    service.log_event("prediction", "USER_01", {"val": 1}, date_str=today_str)
    service.log_event("approval", "USER_02", {"val": 2}, date_str=today_str)
    service.log_event("export", "USER_03", {"val": 3}, date_str=today_str)

    log_filepath = os.path.join(audit_dir, f"{today_str}.jsonl")

    # Delete line 1 (middle event)
    with open(log_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    del lines[1]  # Delete middle line

    with open(log_filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Chain verification MUST fail due to broken previous_event_hash chain
    is_valid = verify_chain(date_str=today_str, audit_dir=audit_dir)
    assert is_valid is False
