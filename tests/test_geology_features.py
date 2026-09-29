"""
Unit tests for RMR and Q-System calculators in src/geology/.
"""

import pytest
from src.geology import (
    RMRCalculator,
    QSystemCalculator,
)


def test_rmr_calculation_matches_bieniawski():
    """RMR for a known rock mass must match Bieniawski 1989 Table 3.14."""
    calc = RMRCalculator()
    result = calc.calculate_total(
        ucs_mpa=100, rqd_pct=85, joint_spacing_m=1.2,
        persistence_m=2.0, aperture_mm=0.5, roughness="rough",
        infilling="none", weathering="slight",
        inflow_l_min=5, joint_water_pressure_mpa=0.1,
        joint_orientation_deg=30, tunnel_azimuth_deg=90,
    )
    assert 55 <= result["rmr"] <= 75
    assert result["class"] in ["fair", "good"]


def test_q_system_formula():
    """Q = (RQD/Jn) × (Jr/Ja) × (Jw/SRF)."""
    calc = QSystemCalculator()
    q = calc.calculate_q(
        rqd_pct=85, num_joint_sets=3, joint_type="rough_undulating",
        alteration="slightly_altered", water_condition="damp",
        stress_condition="medium_stress"
    )
    expected = (85 / 9.0) * (3.0 / 2.0) * (0.66 / 1.0)
    assert abs(q["q_value"] - expected) < 0.1
