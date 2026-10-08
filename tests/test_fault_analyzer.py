"""
Unit tests for 3D structural fault risk classification.
"""

import pytest
from src.geology import classify_fault_risk


def test_classify_fault_risk_high():
    """Test steep dip and close distance yields HIGH risk."""
    risk, reasoning = classify_fault_risk(dip_deg=70.0, distance_m=60.0)
    assert risk == "HIGH"
    assert "steep" in reasoning
    assert "close to blast" in reasoning


def test_classify_fault_risk_medium():
    """Test moderate dip and distance yields MEDIUM risk."""
    risk, reasoning = classify_fault_risk(dip_deg=85.0, distance_m=120.0)
    assert risk == "MEDIUM"
    assert "very steep" in reasoning
    assert "moderate distance" in reasoning


def test_classify_fault_risk_low():
    """Test far distance yields LOW risk."""
    risk, reasoning = classify_fault_risk(dip_deg=70.0, distance_m=300.0)
    assert risk == "LOW"
    assert "far from blast" in reasoning
