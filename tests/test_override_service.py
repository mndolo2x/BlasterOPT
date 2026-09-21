"""
Unit tests for supervisor constraint override service (src/services/override_service.py).
"""

import pytest
from datetime import datetime, timedelta, timezone
from src.services.override_service import (
    ConstraintOverride,
    authorize_override,
    is_override_active,
)


def test_regulatory_limit_override_blocked():
    # Attempting to relax regulatory limit (PPV) must raise ValueError
    with pytest.raises(ValueError) as exc_info:
        authorize_override(
            design_id="PATTERN_001",
            constraint_name="max_ppv_mms",
            original_limit=10.0,
            relaxed_limit=15.0,
            requester_id="ENGINEER_ALICE",
            approver_id="BLASTER_BOB",
            supervisor_id="SUPERVISOR_CHARLIE",
            supervisor_role="PIT_SUPERVISOR",
            reason="Ground near pit boundary needs higher tolerance.",
        )

    assert "REGULATORY LIMIT PROTECTION VIOLATION" in str(exc_info.value)


def test_supervisor_role_enforced():
    # Non-supervisor role attempts override
    with pytest.raises(PermissionError) as exc_info:
        authorize_override(
            design_id="PATTERN_002",
            constraint_name="burden_max_m",
            original_limit=7.0,
            relaxed_limit=8.0,
            requester_id="ENGINEER_ALICE",
            approver_id="BLASTER_BOB",
            supervisor_id="OPERATOR_DAVE",
            supervisor_role="JUNIOR_OPERATOR",
            reason="Increase burden to match available drilling grid.",
        )

    assert "UNAUTHORIZED" in str(exc_info.value)


def test_separation_of_duties_enforced():
    # Supervisor attempts override on design where they are the requester
    with pytest.raises(ValueError) as exc_info:
        authorize_override(
            design_id="PATTERN_003",
            constraint_name="burden_max_m",
            original_limit=7.0,
            relaxed_limit=8.0,
            requester_id="SUPERVISOR_CHARLIE",
            approver_id="BLASTER_BOB",
            supervisor_id="SUPERVISOR_CHARLIE",
            supervisor_role="PIT_SUPERVISOR",
            reason="Adjust burden for wide bench spacing.",
        )

    assert "SEPARATION OF DUTIES VIOLATION" in str(exc_info.value)


def test_valid_override_authorization_and_expiration(tmp_path, monkeypatch):
    test_overrides_path = str(tmp_path / "overrides.json")
    test_audit_dir = str(tmp_path / "audit")
    monkeypatch.setattr("src.services.override_service.OVERRIDES_FILE_PATH", test_overrides_path)
    monkeypatch.setattr("src.services.audit_service.AUDIT_DIR", test_audit_dir)

    override = authorize_override(
        design_id="PATTERN_004",
        constraint_name="burden_max_m",
        original_limit=7.0,
        relaxed_limit=8.0,
        requester_id="ENGINEER_ALICE",
        approver_id="BLASTER_BOB",
        supervisor_id="SUPERVISOR_CHARLIE",
        supervisor_role="PIT_SUPERVISOR",
        reason="Field MWD identified soft sandstone strata allowing larger burden.",
    )

    assert override.override_id is not None
    assert override.constraint_name == "burden_max_m"
    assert override.relaxed_limit == 8.0
    assert is_override_active(override) is True

    # Test expired override
    expired_override = override.model_copy()
    expired_override.expires_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert is_override_active(expired_override) is False
