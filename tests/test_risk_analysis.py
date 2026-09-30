"""
Unit tests for BlastRiskAnalyzer class and risk matrix calculation.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
from src.safety.risk_analysis import BlastRiskAnalyzer, calculate_risk_matrix


def test_blast_risk_analyzer_individual_assessments():
    analyzer = BlastRiskAnalyzer()
    params = {
        "burden_m": 1.0,
        "hole_diameter_m": 0.165,
        "stemming_m": 1.0,
        "has_free_face": False,
        "is_wet": True,
        "explosive_type": "ANFO",
        "closest_receptor_dist_m": 400.0,
        "max_charge_per_delay_kg": 200.0,
        "explosive_mass_kg": 1000.0,
    }

    fly = analyzer.assess_flyrock_risk(params)
    assert fly["likelihood"] >= 4
    assert fly["consequence"] >= 4
    assert fly["risk_score"] >= 16
    assert fly["risk_level"] in ["high", "critical"]

    vib = analyzer.assess_vibration_risk(params)
    assert 1 <= vib["likelihood"] <= 5

    air = analyzer.assess_airblast_risk(params)
    assert 1 <= air["likelihood"] <= 5

    dust = analyzer.assess_dust_risk(params)
    assert 1 <= dust["likelihood"] <= 5

    gas = analyzer.assess_gas_risk(params)
    assert 1 <= gas["likelihood"] <= 5

    misfire = analyzer.assess_misfire_risk(params)
    assert 1 <= misfire["likelihood"] <= 5


def test_blast_risk_analyzer_overall_risk():
    analyzer = BlastRiskAnalyzer()
    params = {
        "burden_m": 3.5,
        "hole_diameter_m": 0.165,
        "closest_receptor_dist_m": 800.0,
        "max_charge_per_delay_kg": 150.0,
    }

    res = analyzer.calculate_overall_risk(params)
    assert "overall_risk_level" in res
    assert "highest_risk_category" in res
    assert isinstance(res["risk_matrix"], pd.DataFrame)
    assert res["risk_matrix"].shape == (5, 5)
    assert isinstance(res["critical_items"], list)
    assert isinstance(res["priority_mitigations"], list)


def test_blast_risk_analyzer_plot():
    analyzer = BlastRiskAnalyzer()
    params = {"closest_receptor_dist_m": 500.0}
    fig = analyzer.generate_risk_matrix_plot(params)
    assert isinstance(fig, go.Figure)


def test_calculate_risk_matrix_legacy():
    res = calculate_risk_matrix(
        ppv_mms=12.0,
        airblast_dbl=125.0,
        flyrock_m=300.0,
        hole_collisions_count=1,
        backbreak_m=2.0,
    )
    assert res["risk_level"] == "extreme"
    assert len(res["hazards"]) >= 3
