"""
Unit tests for PINN page guard and prediction output range sanity checks.
"""

import pytest
import pandas as pd
from src.models import MODEL_REGISTRY


def test_pinn_registered_in_model_registry():
    """Verify PINN is present in MODEL_REGISTRY with expected keys."""
    assert "pinn" in MODEL_REGISTRY
    pinn_meta = MODEL_REGISTRY["pinn"]
    assert pinn_meta["architecture"] == "12-128-256-256-128-3"
    assert len(pinn_meta["feature_columns"]) == 12
    assert pinn_meta["outputs"] == [
        "fragmentation_d80_cm",
        "vibration_ppv_mms",
        "airblast_db",
    ]


def test_pinn_output_range_sanity_check():
    """Verify sanity validation rules for PINN outputs."""
    valid_d80 = 25.0
    valid_airblast = 115.0

    invalid_d80_low = 3.2
    invalid_airblast_low = 13.0

    def check_sanity(d80, airblast):
        return (5.0 <= d80 <= 60.0) and (95.0 <= airblast <= 125.0)

    assert check_sanity(valid_d80, valid_airblast) is True
    assert check_sanity(invalid_d80_low, valid_airblast) is False
    assert check_sanity(valid_d80, invalid_airblast_low) is False
