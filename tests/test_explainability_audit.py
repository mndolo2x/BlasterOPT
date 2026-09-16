"""
Unit tests for explainability audit logging module (src/explainability_audit.py).
"""

import os
import tempfile
import pytest
from src.explainability_audit import log_explanation, get_recent_explanations


def test_log_explanation_inserts_row():
    """Verifies that log_explanation inserts a row into the SQLite database and get_recent_explanations retrieves it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_explainability_audit.db")

        # 1. Log explanation
        res = log_explanation(
            prediction_id="PRED_TEST_101",
            shap_values={"powder_factor_kg_m3": -5.2, "burden_m": 3.1},
            lime_weights={"powder_factor_kg_m3": -0.12},
            natural_language="Ground PPV is within bounds at 6.2 mm/s.",
            user_id="ENGINEER_BOTSWANA_01",
            db_path=db_path,
        )

        assert res["status"] == "success"
        assert res["row_id"] > 0
        assert res["prediction_id"] == "PRED_TEST_101"

        # 2. Retrieve recent explanations
        logs = get_recent_explanations(limit=10, db_path=db_path)
        assert len(logs) == 1
        assert logs[0]["prediction_id"] == "PRED_TEST_101"
        assert logs[0]["user_id"] == "ENGINEER_BOTSWANA_01"
        assert "Ground PPV is within bounds" in logs[0]["natural_language"]
