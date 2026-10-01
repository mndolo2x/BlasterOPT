"""
Unit tests for src/safety.py.
"""
import pytest
from src.safety import (
    NoiseModel,
    DustModel,
    GasModel,
    EnvironmentalAssessment,
    BlastRiskAnalyzer,
)


def test_noise_model():
    res = NoiseModel.predict_noise_level(charge_mass_per_delay_kg=640.0, distance_m=450.0)
    assert 30.0 <= res["dba_peak"] <= 120.0
    assert 40.0 <= res["dbl_peak"] <= 140.0

    df_curve = NoiseModel.calculate_noise_attenuation_curve(charge_mass_per_delay_kg=640.0)
    assert not df_curve.empty
    assert "distance_m" in df_curve.columns


def test_dust_model():
    emissions = DustModel.calculate_dust_emissions(total_explosive_mass_kg=15000.0)
    assert emissions["pm10_mass_kg"] > 0.0
    assert emissions["pm2_5_mass_kg"] > 0.0

    df_disp = DustModel.calculate_dust_dispersion_profile(pm10_mass_kg=emissions["pm10_mass_kg"])
    assert not df_disp.empty
    assert "pm10_concentration_mg_m3" in df_disp.columns


def test_gas_model():
    gases = GasModel.predict_toxic_gases(total_explosive_mass_kg=15000.0, explosive_type="ANFO")
    assert gases["co_volume_liters"] > 0.0
    assert gases["nox_volume_liters"] > 0.0
    assert gases["co2_emissions_kg"] > 0.0


def test_environmental_assessment():
    eia = EnvironmentalAssessment.evaluate_impacts(
        ppv_mms=8.5,
        airblast_dbl=115.0,
        flyrock_m=120.0,
        dust_pm10_kg=100.0,
        co2_mass_kg=500.0,
    )
    assert 1.0 <= eia["overall_impact_rating"] <= 5.0
    assert "is_compliant" in eia


def test_risk_analyzer():
    df_risk = BlastRiskAnalyzer.evaluate_risk_matrix(
        ppv_mms=8.5,
        airblast_dbl=115.0,
        flyrock_m=120.0,
        stemming_m=5.0,
        burden_m=6.0,
    )
    assert not df_risk.empty
    assert "Risk Score" in df_risk.columns
    assert "Risk Level" in df_risk.columns
