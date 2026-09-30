"""
Unit tests for DustModel in src/safety/dust.py.
"""

import pytest
from src.safety.dust import DustModel, predict_dust_dispersion


def test_dust_model():
    model = DustModel(explosive_type="ANFO", charge_mass_kg=1000.0)

    dm = model.calculate_dust_mass(total_explosive_kg=1000.0)
    assert "dust_mass_kg" in dm
    assert dm["dust_mass_kg"] == 220.0  # 1000 * 0.22

    dv = model.calculate_dust_volume(charge_length_m=10.0)
    assert "dust_volume_m3" in dv
    assert dv["dust_volume_m3"] > 0.0

    cloud = model.calculate_dust_cloud_radius(charge_mass_kg=1000.0, wind_speed_m_s=3.0)
    assert "radius_m" in cloud
    assert cloud["radius_m"] > 0.0

    pm10 = model.calculate_pm10_concentration(dust_mass_kg=220.0, distance_m=300.0, wind_speed_m_s=3.0)
    assert "pm10_concentration_mg_m3" in pm10
    assert "exceeds_limit" in pm10


def test_predict_dust_dispersion_wrapper():
    res = predict_dust_dispersion(total_explosive_mass_kg=1000.0, wind_speed_m_s=3.0, distance_m=300.0)
    assert "emission_pm10_kg" in res
    assert "concentration_ug_m3" in res
