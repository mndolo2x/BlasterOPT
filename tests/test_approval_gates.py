"""
Unit tests for human approval gates and blaster sign-off workflow (src/domain/approval.py).
"""

import pytest
from datetime import datetime, timezone
from src.domain.safety_checks import SafetyReport
from src.domain.approval import (
    ApprovalRequest,
    ApprovalDecision,
    approve_design,
    check_approval_gate,
    get_approval_record,
)


def test_approval_request_model():
    safety_rep = SafetyReport(
        overall_status="SAFE",
        checks=[],
        requires_engineer_review=False,
        blocks_export=False,
    )
    req = ApprovalRequest(
        design_id="DESIGN_001",
        design_snapshot={"burden_m": 6.0, "spacing_m": 7.0, "powder_factor_kg_m3": 0.65},
        safety_report=safety_rep,
        requested_by="ENGINEER_01",
        requested_at=datetime.now(timezone.utc).isoformat(),
        model_version="v1.2.0",
        constraint_version="v1.0.0",
        dataset_version="v2.0.0",
    )

    assert req.design_id == "DESIGN_001"
    assert req.design_snapshot["burden_m"] == 6.0
    assert req.requested_by == "ENGINEER_01"
    assert req.model_version == "v1.2.0"


def test_approval_decision_model():
    dec = ApprovalDecision(
        design_id="DESIGN_002",
        decision="CHANGES_REQUESTED",
        decided_by="BLASTER_BW_902",
        decided_at=datetime.now(timezone.utc).isoformat(),
        reason="Increase stemming length on back row by 0.5m.",
        signature="sig_hash_stub_12345",
        conditions=["Add electronic delay buffer of 25ms"],
    )

    assert dec.design_id == "DESIGN_002"
    assert dec.decision == "CHANGES_REQUESTED"
    assert dec.decided_by == "BLASTER_BW_902"
    assert len(dec.conditions) == 1
    assert dec.conditions[0] == "Add electronic delay buffer of 25ms"


def test_approve_safe_design(tmp_path, monkeypatch):
    test_store_path = str(tmp_path / "test_approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_store_path)

    design_id = "PATTERN_TEST_001"
    blaster_id = "BLASTER_BW_101"
    role = "Certified Blaster"

    safety_rep = SafetyReport(
        overall_status="SAFE",
        checks=[],
        requires_engineer_review=False,
        blocks_export=False,
    )

    record = approve_design(
        design_id=design_id,
        blaster_id=blaster_id,
        role=role,
        decision="APPROVED",
        safety_report=safety_rep,
    )

    assert record.design_id == design_id
    assert record.decision == "APPROVED"
    assert record.signature is not None

    is_approved, msg = check_approval_gate(design_id)
    assert is_approved is True
    assert "APPROVED" in msg


def test_approve_unsafe_design_blocked(tmp_path, monkeypatch):
    test_store_path = str(tmp_path / "test_approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_store_path)

    design_id = "PATTERN_UNSAFE_002"
    safety_rep = SafetyReport(
        overall_status="UNSAFE",
        checks=[],
        requires_engineer_review=True,
        blocks_export=True,
    )

    with pytest.raises(ValueError) as exc_info:
        approve_design(
            design_id=design_id,
            blaster_id="BLASTER_BW_101",
            role="Certified Blaster",
            decision="APPROVED",
            safety_report=safety_rep,
        )

    assert "CANNOT APPROVE INFEASIBLE / UNSAFE DESIGN" in str(exc_info.value)


def test_approve_requires_review_mandatory_override_reasoning(tmp_path, monkeypatch):
    test_store_path = str(tmp_path / "test_approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_store_path)

    design_id = "PATTERN_WARN_003"
    safety_rep = SafetyReport(
        overall_status="REQUIRES_REVIEW",
        checks=[],
        requires_engineer_review=True,
        blocks_export=False,
    )

    # Attempting to approve without override reasoning must raise ValueError
    with pytest.raises(ValueError) as exc_info:
        approve_design(
            design_id=design_id,
            blaster_id="BLASTER_BW_101",
            role="Certified Blaster",
            decision="APPROVED",
            override_reasoning="",
            safety_report=safety_rep,
        )

    assert "MANDATORY OVERRIDE REASONING REQUIRED" in str(exc_info.value)

    # Approving with valid override reasoning succeeds
    rec = approve_design(
        design_id=design_id,
        blaster_id="BLASTER_BW_101",
        role="Certified Blaster",
        decision="APPROVED",
        override_reasoning="Applied 25ms delay buffer on row 3 to prevent PPV spike.",
        safety_report=safety_rep,
    )
    assert rec.decision == "APPROVED"
    assert rec.reason is not None


def test_check_approval_gate_unapproved(tmp_path, monkeypatch):
    test_store_path = str(tmp_path / "test_approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_store_path)

    is_approved, msg = check_approval_gate("NON_EXISTENT_PATTERN")
    assert is_approved is False
    assert "APPROVAL REQUIRED" in msg


def test_check_approval_gate_changes_requested(tmp_path, monkeypatch):
    test_store_path = str(tmp_path / "test_approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_store_path)

    design_id = "PATTERN_CHANGES_004"
    approve_design(
        design_id=design_id,
        blaster_id="BLASTER_BW_101",
        decision="CHANGES_REQUESTED",
        override_reasoning="Stemming length must be increased to 5.5m.",
    )

    is_approved, msg = check_approval_gate(design_id)
    assert is_approved is False
    assert "CHANGES REQUESTED" in msg
