"""
Unit tests for Botswana Regulatory Compliance module.
"""

import os
import pytest
from src.regulatory import load_regulatory_limits, check_compliance, generate_compliance_report


def test_load_regulatory_limits_returns_dict():
    """Test load_regulatory_limits returns active config dictionary."""
    limits = load_regulatory_limits()

    assert isinstance(limits, dict)
    assert "max_ppv_mms" in limits
    assert "max_airblast_dbl" in limits
    assert "max_flyrock_m" in limits
    assert limits["max_ppv_mms"] == 10.0


def test_check_compliance_compliant_design():
    """Test check_compliance returns True and empty violations for safe design."""
    params = {"stemming_m": 5.0, "powder_factor_kg_m3": 0.65}
    preds = {"ppv_mms": 6.5, "airblast_dbl": 112.0, "flyrock_m": 110.0}

    res = check_compliance(params, preds)

    assert isinstance(res, dict)
    assert res["is_compliant"] is True
    assert len(res["violations"]) == 0


def test_check_compliance_returns_correct_violations():
    """Test check_compliance returns False and populated violations when thresholds are exceeded."""
    params = {"stemming_m": 1.5}  # Stemming < 2.5m min limit
    preds = {
        "ppv_mms": 14.2,         # PPV > 10.0 mm/s limit
        "airblast_dbl": 128.0,   # Airblast > 120 dBL limit
        "flyrock_m": 310.0,      # Flyrock > 250 m limit
    }

    res = check_compliance(params, preds)

    assert res["is_compliant"] is False
    assert len(res["violations"]) == 4
    assert any("Ground Vibration" in v for v in res["violations"])
    assert any("Airblast Overpressure" in v for v in res["violations"])
    assert any("Flyrock Range" in v for v in res["violations"])
    assert any("Stemming Length" in v for v in res["violations"])
    assert len(res["recommendations"]) > 0


def test_generate_compliance_report_creates_non_zero_file(tmp_path):
    """Test generate_compliance_report creates a non-empty PDF file."""
    output_pdf = str(tmp_path / "test_compliance.pdf")
    params = {"stemming_m": 4.5, "powder_factor_kg_m3": 0.6}
    preds = {"ppv_mms": 8.0, "airblast_dbl": 114.0, "flyrock_m": 120.0}

    generated_path = generate_compliance_report(params, preds, output_path=output_pdf)

    assert os.path.exists(generated_path)
    assert os.path.getsize(generated_path) > 0, "Expected non-zero PDF file size"
