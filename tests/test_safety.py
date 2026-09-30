"""
Unit tests for src/safety/ package modules.
"""

import pytest
from src.safety import (
    DustModel,
    GasModel,
    NoiseModel,
    EnvironmentalAssessment,
    BlastRiskAnalyzer,
    predict_dust_dispersion,
    predict_toxic_gases,
    predict_noise_overpressure,
    evaluate_environmental_impact,
    calculate_risk_matrix,
    SafetyRenderer,
)


def test_dust_mass_calculation():
    """Dust mass must match Tyupin & Bolotova (2026) range."""
    model = DustModel(explosive_type="ANFO", charge_mass_kg=1000)
    result = model.calculate_dust_mass(1000)
    assert 180 <= result["dust_mass_kg"] <= 250


def test_gas_emission_factors():
    """CO and NOx emissions must match emission factor table."""
    model = GasModel(explosive_type="ANFO")
    result = model.calculate_gas_emissions(1000)
    assert abs(result["CO_kg"] - 15.0) < 0.1
    assert abs(result["NOx_kg"] - 8.0) < 0.1


def test_noise_attenuation():
    """SPL must decrease with distance."""
    model = NoiseModel(charge_per_delay_kg=100, depth_of_burial_m=10)
    r1 = model.calculate_noise_at_distance(500)
    r2 = model.calculate_noise_at_distance(1000)
    assert r2["spl_db"] < r1["spl_db"]


def test_risk_matrix_returns_valid_level():
    """Risk level must be one of the defined categories."""
    analyzer = BlastRiskAnalyzer()
    params = {"closest_receptor_dist_m": 500.0}
    result = analyzer.calculate_overall_risk(params)
    assert result["overall_risk_level"] in ["low", "moderate", "high", "critical"]


def test_eia_breakdown_has_five_categories():
    """Environmental assessment must score all five categories."""
    receptors = {"village": 800.0}
    params = {"explosive_mass_kg": 500.0}
    eia = EnvironmentalAssessment(blast_params=params, receptor_distances=receptors)
    result = eia.calculate_total_impact()
    assert len(result["breakdown"]) == 5
    assert all(1 <= v <= 5 for v in result["breakdown"].values())


def test_dust_dispersion():
    res = predict_dust_dispersion(total_explosive_mass_kg=2000.0, wind_speed_m_s=3.5, distance_m=300.0)
    assert "emission_pm10_kg" in res
    assert "concentration_ug_m3" in res
    assert res["concentration_ug_m3"] > 0.0


def test_toxic_gases():
    res = predict_toxic_gases(explosive_mass_kg=2000.0, anfo_fuel_oil_pct=6.0)
    assert "co_emissions_liters" in res
    assert "nox_emissions_liters" in res


def test_noise_overpressure():
    res = predict_noise_overpressure(max_charge_per_delay_kg=300.0, distance_m=400.0)
    assert "noise_dbl" in res
    assert "scaled_distance" in res


def test_environmental_impact():
    res = evaluate_environmental_impact(ppv_mms=6.5, airblast_dbl=115.0, flyrock_m=120.0, pm10_ug_m3=80.0, nox_ppm=2.0)
    assert "eia_score" in res
    assert "impact_class" in res


def test_risk_matrix():
    res = calculate_risk_matrix(ppv_mms=12.0, airblast_dbl=125.0, flyrock_m=280.0, hole_collisions_count=1, backbreak_m=3.5)
    assert "overall_risk_score" in res
    assert res["risk_level"] in ("high", "extreme")


def test_safety_renderer():
    renderer = SafetyRenderer()
    fig = renderer.plot_dust_plume(pm10_concentration_ug_m3=180.0, distance_m=300.0)
    assert fig is not None
