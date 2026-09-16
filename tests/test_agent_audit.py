"""
Unit tests for Agent Audit Log module (src/agent/audit.py).
"""

import os
import pytest
from src.agent.audit import (
    log_interaction,
    log_decision,
    get_interaction_history,
    get_decision_history,
    export_audit_log_json,
)


def test_log_interaction_inserts_row():
    """Test log_interaction inserts a row into SQLite database."""
    row_id = log_interaction(
        session_id="SESS_AUDIT_TEST_01",
        user_id="ENG_AUDIT_01",
        user_message="Design blast for bench 14",
        agent_response="Generated design for bench 14",
        tools_called=["design_blast", "predict_vibration"],
        guardrail_trips=[],
    )

    assert row_id > 0

    history = get_interaction_history(session_id="SESS_AUDIT_TEST_01", limit=5)
    assert len(history) >= 1
    assert history[0]["session_id"] == "SESS_AUDIT_TEST_01"
    assert history[0]["user_id"] == "ENG_AUDIT_01"
    assert "bench 14" in history[0]["user_message"]


def test_log_decision_inserts_row():
    """Test log_decision inserts a row into SQLite database."""
    row_id = log_decision(
        session_id="SESS_AUDIT_TEST_02",
        user_id="BLASTER_01",
        design_id="DES_JWA_101",
        decision="OVERRIDE_STEMMING",
        reason_code="HIGH_AIRBLAST_RISK",
        original_value={"stemming_m": 4.5},
        new_value={"stemming_m": 5.5},
    )

    assert row_id > 0

    history = get_decision_history(session_id="SESS_AUDIT_TEST_02", limit=5)
    assert len(history) >= 1
    assert history[0]["session_id"] == "SESS_AUDIT_TEST_02"
    assert history[0]["design_id"] == "DES_JWA_101"
    assert history[0]["reason_code"] == "HIGH_AIRBLAST_RISK"


def test_export_audit_log_json():
    """Test export_audit_log_json generates valid JSON audit export package."""
    export_file = export_audit_log_json("data/processed/test_regulatory_audit_export.json")

    assert os.path.exists(export_file)
    assert os.path.getsize(export_file) > 0
