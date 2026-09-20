"""
Unit tests for ApprovalService (src/services/approval_service.py).
"""

import pytest
from src.domain.safety_checks import SafetyReport
from src.domain.approval import check_approval_gate
from src.services.approval_service import (
    submit_for_approval,
    record_decision,
    get_approval_status,
    ApprovalStatus,
)


def test_submit_for_approval(tmp_path, monkeypatch):
    test_req_path = str(tmp_path / "approval_requests.json")
    test_audit_dir = str(tmp_path / "audit")
    monkeypatch.setattr("src.services.approval_service.REQUESTS_FILE_PATH", test_req_path)
    monkeypatch.setattr("src.services.approval_service.AUDIT_LOG_DIR", test_audit_dir)

    design = {"design_id": "PATTERN_101", "burden_m": 6.0, "spacing_m": 7.0}
    safety_rep = SafetyReport(
        overall_status="SAFE",
        checks=[],
        requires_engineer_review=False,
        blocks_export=False,
    )

    req = submit_for_approval(design, safety_report=safety_rep, user_id="ENGINEER_ALICE")

    assert req.design_id == "PATTERN_101"
    assert req.requested_by == "ENGINEER_ALICE"
    assert req.design_snapshot["burden_m"] == 6.0

    status_obj = get_approval_status("PATTERN_101")
    assert status_obj.status == "PENDING"
    assert status_obj.request is not None


def test_record_decision_role_enforcement(tmp_path, monkeypatch):
    test_req_path = str(tmp_path / "approval_requests.json")
    test_app_path = str(tmp_path / "approvals.json")
    monkeypatch.setattr("src.services.approval_service.REQUESTS_FILE_PATH", test_req_path)
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_app_path)

    design = {"design_id": "PATTERN_102", "burden_m": 6.0}
    safety_rep = SafetyReport(overall_status="SAFE", checks=[], requires_engineer_review=False, blocks_export=False)
    req = submit_for_approval(design, safety_report=safety_rep, user_id="ENGINEER_ALICE")

    # Non-certified blaster attempts to record decision
    with pytest.raises(PermissionError) as exc_info:
        record_decision(
            design_id_or_request=req,
            decision="APPROVED",
            user_id="OPERATOR_BOB",
            user_role="JUNIOR_OPERATOR",
        )

    assert "UNAUTHORIZED" in str(exc_info.value)


def test_separation_of_duties_enforced(tmp_path, monkeypatch):
    test_req_path = str(tmp_path / "approval_requests.json")
    test_app_path = str(tmp_path / "approvals.json")
    monkeypatch.setattr("src.services.approval_service.REQUESTS_FILE_PATH", test_req_path)
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_app_path)

    design = {"design_id": "PATTERN_103", "burden_m": 6.0}
    safety_rep = SafetyReport(overall_status="SAFE", checks=[], requires_engineer_review=False, blocks_export=False)
    req = submit_for_approval(design, safety_report=safety_rep, user_id="BLASTER_CHARLIE")

    # Submitter attempts to self-approve
    with pytest.raises(ValueError) as exc_info:
        record_decision(
            design_id_or_request=req,
            decision="APPROVED",
            user_id="BLASTER_CHARLIE",
            user_role="CERTIFIED_BLASTER",
        )

    assert "SEPARATION OF DUTIES VIOLATION" in str(exc_info.value)


def test_approval_blocked_for_unsafe_design(tmp_path, monkeypatch):
    test_req_path = str(tmp_path / "approval_requests.json")
    test_app_path = str(tmp_path / "approvals.json")
    monkeypatch.setattr("src.services.approval_service.REQUESTS_FILE_PATH", test_req_path)
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_app_path)

    design = {"design_id": "PATTERN_104", "burden_m": 6.0}
    safety_rep = SafetyReport(overall_status="UNSAFE", checks=[], requires_engineer_review=True, blocks_export=True)
    req = submit_for_approval(design, safety_report=safety_rep, user_id="ENGINEER_ALICE")

    # Certified blaster attempts to approve UNSAFE design
    with pytest.raises(ValueError) as exc_info:
        record_decision(
            design_id_or_request=req,
            decision="APPROVED",
            user_id="BLASTER_DAVE",
            user_role="CERTIFIED_BLASTER",
        )

    assert "CANNOT APPROVE INFEASIBLE / UNSAFE DESIGN" in str(exc_info.value)


def test_export_blocked_without_approval(tmp_path, monkeypatch):
    test_app_path = str(tmp_path / "approvals.json")
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_app_path)

    is_approved, msg = check_approval_gate("PATTERN_UNREVIEWED_999", action="export")
    assert is_approved is False
    assert "APPROVAL REQUIRED" in msg


def test_export_allowed_after_approval(tmp_path, monkeypatch):
    test_req_path = str(tmp_path / "approval_requests.json")
    test_app_path = str(tmp_path / "approvals.json")
    monkeypatch.setattr("src.services.approval_service.REQUESTS_FILE_PATH", test_req_path)
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_app_path)

    design = {"design_id": "PATTERN_APPROVED_888", "burden_m": 6.0}
    safety_rep = SafetyReport(overall_status="SAFE", checks=[], requires_engineer_review=False, blocks_export=False)
    req = submit_for_approval(design, safety_report=safety_rep, user_id="ENGINEER_ALICE")

    record_decision(
        design_id_or_request=req,
        decision="APPROVED",
        user_id="BLASTER_DAVE",
        user_role="CERTIFIED_BLASTER",
    )

    is_approved, msg = check_approval_gate("PATTERN_APPROVED_888", action="export")
    assert is_approved is True
    assert "APPROVED" in msg


def test_record_decision_uncertainty_acknowledgment_required(tmp_path, monkeypatch):
    test_req_path = str(tmp_path / "approval_requests.json")
    test_app_path = str(tmp_path / "approvals.json")
    monkeypatch.setattr("src.services.approval_service.REQUESTS_FILE_PATH", test_req_path)
    monkeypatch.setattr("src.domain.approval.APPROVALS_FILE_PATH", test_app_path)

    design = {"design_id": "PATTERN_105", "burden_m": 6.0}
    safety_rep = SafetyReport(overall_status="REQUIRES_REVIEW", checks=[], requires_engineer_review=True, blocks_export=False)
    req = submit_for_approval(design, safety_report=safety_rep, user_id="ENGINEER_ALICE")

    # Approving without acknowledging uncertainty must raise ValueError
    with pytest.raises(ValueError) as exc_info:
        record_decision(
            design_id_or_request=req,
            decision="APPROVED",
            user_id="BLASTER_DAVE",
            user_role="CERTIFIED_BLASTER",
            acknowledged_uncertainty=False,
        )

    assert "UNCERTAINTY ACKNOWLEDGMENT REQUIRED" in str(exc_info.value)

    # Approving with acknowledged_uncertainty=True succeeds
    dec = record_decision(
        design_id_or_request=req,
        decision="APPROVED",
        user_id="BLASTER_DAVE",
        user_role="CERTIFIED_BLASTER",
        acknowledged_uncertainty=True,
        reason="Reviewed 95% confidence bounds and accepted risk profile.",
    )

    assert dec.decision == "APPROVED"
    assert dec.decided_by == "BLASTER_DAVE"

    status_obj = get_approval_status("PATTERN_105")
    assert status_obj.status == "APPROVED"
