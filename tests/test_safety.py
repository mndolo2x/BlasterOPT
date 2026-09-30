"""
Unit tests for src/safety/ package modules.
"""

import pytest
from src.safety import (
    predict_dust_dispersion,
    predict_toxic_gases,
    predict_noise_overpressure,
    evaluate_environmental_impact,
    calculate_risk_matrix,
    SafetyRenderer,
)


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
