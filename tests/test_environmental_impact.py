"""
Unit tests for EnvironmentalAssessment class.
"""

import os
import pytest
from src.safety.environmental_impact import EnvironmentalAssessment, evaluate_environmental_impact


def test_environmental_assessment_initialization():
    receptors = {"village": 800.0, "road": 450.0}
    params = {"explosive_mass_kg": 1000.0, "max_charge_per_delay_kg": 200.0}
    ea = EnvironmentalAssessment(blast_params=params, receptor_distances=receptors)
    assert ea.closest_distance_m == 450.0


def test_environmental_assessment_invalid_inputs():
    with pytest.raises(ValueError):
        EnvironmentalAssessment(blast_params={}, receptor_distances={})
    with pytest.raises(ValueError):
        EnvironmentalAssessment(blast_params={}, receptor_distances={"site": -100.0})


def test_environmental_assessment_categories():
    receptors = {"camp": 500.0}
    params = {
        "explosive_mass_kg": 500.0,
        "max_charge_per_delay_kg": 100.0,
        "depth_of_burial_m": 3.0,
        "burden_m": 3.5,
        "hole_diameter_m": 0.165,
    }
    ea = EnvironmentalAssessment(blast_params=params, receptor_distances=receptors)

    dust = ea.assess_dust()
    assert 1 <= dust["impact_score"] <= 5

    gas = ea.assess_gas()
    assert 1 <= gas["impact_score"] <= 5

    noise = ea.assess_noise()
    assert 1 <= noise["impact_score"] <= 5

    vib = ea.assess_vibration()
    assert 1 <= vib["impact_score"] <= 5

    fly = ea.assess_flyrock()
    assert 1 <= fly["impact_score"] <= 5


def test_environmental_assessment_total_impact():
    receptors = {"village": 800.0, "road": 450.0}
    params = {"explosive_mass_kg": 1000.0, "max_charge_per_delay_kg": 200.0}
    ea = EnvironmentalAssessment(blast_params=params, receptor_distances=receptors)

    tot = ea.calculate_total_impact()
    assert 5 <= tot["total_score"] <= 25
    assert tot["overall_rating"] in ["low", "moderate", "high", "severe"]
    assert "breakdown" in tot
    assert isinstance(tot["priority_mitigations"], list)


def test_environmental_assessment_generate_report(tmp_path):
    receptors = {"village": 800.0}
    params = {"explosive_mass_kg": 500.0}
    ea = EnvironmentalAssessment(blast_params=params, receptor_distances=receptors)

    report_file = tmp_path / "eia_test.pdf"
    file_path = ea.generate_eia_report(str(report_file))
    assert os.path.exists(file_path)


def test_evaluate_environmental_impact_legacy():
    res = evaluate_environmental_impact(
        ppv_mms=5.0,
        airblast_dbl=115.0,
        flyrock_m=100.0,
        pm10_ug_m3=40.0,
        nox_ppm=1.0,
    )
    assert "eia_score" in res
    assert "impact_class" in res
    assert "sub_scores" in res
