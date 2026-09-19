"""
Unit tests for human approval gates and blaster sign-off workflow (src/domain/approval.py).
"""

import pytest
from src.domain.safety_checks import SafetyReport, SafetyCheck
from src.domain.approval import approve_design, check_approval_gate, get_approval_record


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
    assert record.signature_hash is not None

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
    assert rec.override_reasoning is not None


def test_check_approval_gate_unapproved(tmp_path, monkeypatch):
    test_store_path = str(tmp_path / "test_approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_store_path)

    is_approved, msg = check_approval_gate("NON_EXISTENT_PATTERN")
    assert is_approved is False
    assert "APPROVAL REQUIRED" in msg
