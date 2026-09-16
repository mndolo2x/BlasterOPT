"""
Unit tests for Explainability Audit Logging module (src/explainability_audit.py).
"""

import os
import tempfile
import pytest
from src.explainability_audit import log_explanation, get_explanation_history, get_recent_explanations


def test_log_explanation_inserts_row_into_database():
    """Test log_explanation inserts an append-only row into the SQLite audit database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_audit.db")

        res = log_explanation(
            prediction_id="PRED_JWA_2026_01",
            shap_values={"powder_factor_kg_m3": -8.5, "max_charge_per_delay_kg": 14.2},
            lime_weights={"powder_factor_kg_m3": -0.10, "max_charge_per_delay_kg": 0.18},
            natural_language="Ground PPV is predicted to meet safety bounds at 6.5 mm/s.",
            user_id="BLASTER_JWA_01",
            db_path=db_path,
        )

        assert res["status"] == "success"
        assert res["row_id"] == 1
        assert res["prediction_id"] == "PRED_JWA_2026_01"


def test_get_explanation_history_returns_correct_rows():
    """Test get_explanation_history retrieves logged explanation rows with optional filtering."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_audit.db")

        log_explanation(
            prediction_id="PRED_JWA_2026_01",
            shap_values={"powder_factor": -5.0},
            lime_weights={"powder_factor": -0.1},
            natural_language="Prediction #1 explanation",
            user_id="USER_A",
            db_path=db_path,
        )

        log_explanation(
            prediction_id="PRED_ORA_2026_02",
            shap_values={"burden": 2.5},
            lime_weights={"burden": 0.08},
            natural_language="Prediction #2 explanation",
            user_id="USER_B",
            db_path=db_path,
        )

        # Retrieve all history
        all_logs = get_explanation_history(limit=10, db_path=db_path)
        assert len(all_logs) == 2

        # Filter by prediction_id
        filtered_pred = get_explanation_history(prediction_id="PRED_JWA_2026_01", db_path=db_path)
        assert len(filtered_pred) == 1
        assert filtered_pred[0]["prediction_id"] == "PRED_JWA_2026_01"
        assert filtered_pred[0]["user_id"] == "USER_A"

        # Filter by user_id
        filtered_user = get_explanation_history(user_id="USER_B", db_path=db_path)
        assert len(filtered_user) == 1
        assert filtered_user[0]["prediction_id"] == "PRED_ORA_2026_02"
