"""
Unit tests for uncertainty-aware safety checks module (src/domain/safety_checks.py).
"""

import pytest
from src.domain.safety_checks import evaluate_safety, SafetyCheck, SafetyReport


def test_safety_check_safe_when_upper_ci_below_limit():
    predictions = {"ppv_mms": 4.0, "airblast_dbl": 110.0, "flyrock_m": 80.0}
    limits = {"max_ppv_mms": 10.0, "max_airblast_dbl": 120.0, "max_flyrock_m": 250.0}
    confidence_intervals = {
        "ppv_mms": (3.2, 4.8),
        "airblast_dbl": (105.0, 115.0),
        "flyrock_m": (70.0, 95.0),
    }

    report = evaluate_safety(predictions, limits=limits, confidence_intervals=confidence_intervals)

    assert report.overall_status == "SAFE"
    assert report.requires_engineer_review is False
    assert report.blocks_export is False
    assert len(report.checks) == 3


def test_safety_check_requires_review_when_upper_ci_exceeds_limit():
    # Mean is 8.5 mm/s <= 10.0 limit, but upper 95 bound is 11.2 > 10.0 limit
    predictions = {"ppv_mms": 8.5, "airblast_dbl": 110.0}
    limits = {"max_ppv_mms": 10.0, "max_airblast_dbl": 120.0}
    confidence_intervals = {
        "ppv_mms": (6.0, 11.2),
        "airblast_dbl": (105.0, 115.0),
    }

    report = evaluate_safety(predictions, limits=limits, confidence_intervals=confidence_intervals)

    assert report.overall_status == "REQUIRES_REVIEW"
    assert report.requires_engineer_review is True
    assert report.blocks_export is False

    ppv_check = next(c for c in report.checks if c.check_name == "ppv_limit")
    assert ppv_check.status == "REQUIRES_REVIEW"
    assert "95% upper confidence bound" in ppv_check.reasoning


def test_safety_check_unsafe_when_prediction_exceeds_limit():
    # Mean is 12.5 mm/s > 10.0 limit
    predictions = {"ppv_mms": 12.5, "airblast_dbl": 110.0}
    limits = {"max_ppv_mms": 10.0, "max_airblast_dbl": 120.0}

    report = evaluate_safety(predictions, limits=limits)

    assert report.overall_status == "UNSAFE"
    assert report.requires_engineer_review is True
    assert report.blocks_export is True

    ppv_check = next(c for c in report.checks if c.check_name == "ppv_limit")
    assert ppv_check.status == "UNSAFE"


def test_safety_check_stemming_confinement():
    blast_params = {"stemming_m": 2.0} # Below 2.5 min stemming limit
    predictions = {"ppv_mms": 5.0}

    report = evaluate_safety(predictions, blast_params=blast_params)

    assert report.overall_status == "UNSAFE"
    stem_check = next(c for c in report.checks if c.check_name == "stemming_confinement")
    assert stem_check.status == "UNSAFE"
    assert "below minimum confinement limit" in stem_check.reasoning
