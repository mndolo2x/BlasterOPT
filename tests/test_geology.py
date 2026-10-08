"""
Unit tests for src/geology.py.
"""
import pytest
from src.geology import RMRCalculator, QSystemCalculator, JointAnalyzer, FaultStructureModel, OreBodyModel


def test_rmr_calculator():
    res = RMRCalculator.calculate_rmr(
        ucs_mpa=100.0,
        rqd_pct=85.0,
        spacing_m=1.2,
        persistence="1-3 m",
        aperture="0.1-1.0 mm",
        roughness="Rough",
        infilling="None",
        weathering="Slightly weathered",
        groundwater="Damp",
        orientation="Fair",
    )
    # UCS(12) + RQD(17) + Spacing(15) + Condition(30) + Groundwater(10) + Orientation(-5) = 79
    assert res["rmr_score"] == 79
    assert res["strength_rating"] == 12
    assert res["rqd_rating"] == 17
    assert res["spacing_rating"] == 15
    assert res["condition_rating"] == 30
    assert res["groundwater_rating"] == 10
    assert res["orientation_rating"] == -5
    assert res["rock_class"] == "Good Rock (Class II)"


def test_q_system_calculator():
    res = QSystemCalculator.calculate_q(rqd_pct=75.0, jn=9.0, jr=2.0, ja=1.0)
    assert res["q_value"] > 0.0
    assert "quality_description" in res


def test_joint_analyzer():
    joint_data = [{"dip": 60.0, "dip_direction": 120.0}]
    df = JointAnalyzer.analyze_joint_sets(joint_data)
    assert not df.empty
    assert "dip" in df.columns


def test_fault_model():
    faults = FaultStructureModel.analyze_faults({})
    assert isinstance(faults, list)
    assert len(faults) > 0


def test_ore_body_model():
    res = OreBodyModel.estimate_dilution(powder_factor_kg_m3=0.65)
    assert "dilution_pct" in res
    assert "recovery_pct" in res
