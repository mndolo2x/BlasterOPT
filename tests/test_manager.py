"""
Unit tests for Calibration Manager (src/site_calibration/calibration_manager.py).
"""

import os
import pytest
import numpy as np
from src.site_calibration.models import SeismographReading
from src.site_calibration.calibration_manager import CalibrationManager


def test_manager_end_to_end_workflow(tmp_path):
    """
    Test end-to-end CalibrationManager workflow:
    Ingest readings -> fit_site -> predict_ppv -> get_calibration_quality.
    """
    db_path = str(tmp_path / "test_calibration_db.json")
    manager = CalibrationManager(db_file_path=db_path)

    np.random.seed(42)
    site_id = "SITE_JWANENG_CUT8"
    rock_type = "Kimberlite"

    # Add 25 synthetic readings
    for i in range(1, 26):
        dist = 100.0 + i * 35.0
        charge = 400.0 + (i % 5) * 50.0
        sd = dist / np.sqrt(charge)
        ppv = 550.0 * (sd ** (-1.65)) * np.random.normal(1.0, 0.02)

        reading = SeismographReading(
            blast_id=f"BLAST_{i:03d}",
            site_id=site_id,
            distance_m=dist,
            ppv_mm_s=ppv,
            charge_per_delay_kg=charge,
            dominant_frequency_hz=25.0,
            gsi_of_transmission_strata=55.0 + (i % 15),
            rock_type=rock_type,
        )
        manager.add_reading(reading)

    assert len(manager.readings_db) == 25
    assert site_id in manager.list_sites()

    # Fit site
    cal_res = manager.fit_site(site_id, rock_type=rock_type)
    assert cal_res.fit_quality in ["EXCELLENT", "ACCEPTABLE"]
    assert cal_res.parameters is not None
    assert cal_res.parameters.r_squared >= 0.85

    # Predict PPV
    pred = manager.predict_ppv(
        site_id=site_id,
        distance_m=450.0,
        charge_per_delay_kg=640.0,
        gsi=60.0,
        rock_type=rock_type,
    )

    assert pred.ppv_mm_s > 0
    assert pred.lower_95 < pred.ppv_mm_s < pred.upper_95

    # Check calibration quality report
    quality = manager.get_calibration_quality(site_id, rock_type=rock_type)
    assert quality.is_statistically_sound is True
    assert quality.status in ["EXCELLENT", "ACCEPTABLE"]


def test_manager_uncalibrated_generic_fallback(tmp_path):
    """
    Test manager returning generic USBM fallback prediction when site is uncalibrated.
    """
    db_path = str(tmp_path / "test_uncalibrated_db.json")
    manager = CalibrationManager(db_file_path=db_path)

    pred = manager.predict_ppv(
        site_id="SITE_UNCALIBRATED",
        distance_m=500.0,
        charge_per_delay_kg=500.0,
    )

    assert pred.method == "USBM_GENERIC_FALLBACK"
    assert pred.parameters_used["K"] == 1140.0
    assert pred.parameters_used["B"] == 1.6
