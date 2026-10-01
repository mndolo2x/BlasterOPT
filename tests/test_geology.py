"""
Unit tests for src/geology.py.
"""
import pytest
from src.geology import RMRCalculator, QSystemCalculator, JointAnalyzer, FaultStructureModel, OreBodyModel


def test_rmr_calculator():
    res = RMRCalculator.calculate_rmr(ucs_mpa=120.0, rqd_pct=75.0, spacing_m=0.5)
    assert 0 <= res["rmr_score"] <= 100
    assert "rock_class" in res


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
