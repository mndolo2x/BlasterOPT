"""
Unit tests for GasModel and toxic gas prediction functions.
"""

import pytest
from src.safety.toxic_gases import GasModel, predict_toxic_gases


def test_gas_model_emissions():
    model = GasModel("ANFO")
    emissions = model.calculate_gas_emissions(1000.0)
    assert emissions["CO_kg"] == 15.0
    assert emissions["NOx_kg"] == 8.0
    assert emissions["CO2_kg"] == 180.0
    assert emissions["total_toxic_kg"] == 23.0


def test_gas_model_invalid_type():
    with pytest.raises(ValueError):
        GasModel("InvalidExplosive")


def test_gas_model_concentration_and_reentry():
    model = GasModel("Emulsion")
    emissions = model.calculate_gas_emissions(500.0)
    co_mass = emissions["CO_kg"]

    res = model.calculate_gas_concentration(
        gas_mass_kg=co_mass,
        gas_type="CO",
        ventilation_rate_m3_s=10.0,
        volume_m3=5000.0,
        time_s=300.0,
    )
    assert "concentration_ppm" in res
    assert "time_to_safe_s" in res
    assert isinstance(res["exceeds_limit"], bool)

    reentry = model.calculate_reentry_time(
        gas_mass_kg=co_mass,
        gas_type="CO",
        ventilation_rate_m3_s=10.0,
        volume_m3=5000.0,
    )
    assert reentry >= 0.0


def test_gas_model_ventilation_requirement():
    model = GasModel("ANFO")
    vent_req = model.calculate_ventilation_requirement(
        gas_mass_kg=10.0,
        gas_type="CO",
        target_ppm=50.0,
        time_s=1800.0,
    )
    assert vent_req > 0.0


def test_predict_toxic_gases_legacy():
    result = predict_toxic_gases(explosive_mass_kg=500.0)
    assert "co_emissions_liters" in result
    assert "nox_emissions_liters" in result
    assert "hazard_level" in result
