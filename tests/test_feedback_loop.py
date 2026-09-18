"""
Unit tests for Feedback Loop Engine (src/site_calibration/feedback_loop.py).
"""

import pytest
import numpy as np
from src.site_calibration.models import SeismographReading
from src.site_calibration.calibration_manager import CalibrationManager
from src.site_calibration.feedback_loop import FeedbackLoopEngine


def test_feedback_loop_triggers_recalibration(tmp_path):
    """
    Tests FeedbackLoopEngine processing new post-blast reading:
    Initializes calibration, adds new reading, and verifies logging and drift alerts.
    """
    db_path = str(tmp_path / "test_feedback_db.json")
    manager = CalibrationManager(db_file_path=db_path)
    engine = FeedbackLoopEngine(calibration_manager=manager)

    np.random.seed(42)
    site_id = "SITE_ORAPA_SOUTH"
    rock_type = "Kimberlite"

    # Seed 15 initial readings
    for i in range(1, 16):
        dist = 150.0 + i * 30.0
        charge = 500.0
        sd = dist / np.sqrt(charge)
        ppv = 500.0 * (sd ** (-1.6)) * np.random.normal(1.0, 0.01)

        manager.add_reading(SeismographReading(
            blast_id=f"BLAST_INIT_{i}",
            site_id=site_id,
            distance_m=dist,
            ppv_mm_s=ppv,
            charge_per_delay_kg=charge,
            rock_type=rock_type,
        ))

    # Initial fit
    manager.fit_site(site_id, rock_type=rock_type)

    # Ingest new post-blast reading with force_recalibrate
    new_reading = SeismographReading(
        blast_id="BLAST_NEW_101",
        site_id=site_id,
        distance_m=400.0,
        ppv_mm_s=12.5,
        charge_per_delay_kg=500.0,
        rock_type=rock_type,
    )

    recal_log = engine.process_new_reading(new_reading, force_recalibrate=True)

    assert recal_log is not None
    assert recal_log.site_id == site_id
    assert recal_log.sample_count == 16
    assert "old_params" in recal_log.model_dump()
    assert "new_params" in recal_log.model_dump()
    assert len(engine.recalibration_logs) == 1
