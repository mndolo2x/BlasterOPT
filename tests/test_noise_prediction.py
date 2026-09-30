"""
Unit tests for NoiseModel and airblast noise prediction functions.
"""

import pytest
from src.safety.noise_prediction import NoiseModel, predict_noise_overpressure


def test_noise_model_initialization():
    model = NoiseModel(charge_per_delay_kg=150.0, depth_of_burial_m=3.0)
    assert model.charge_per_delay_kg == 150.0
    assert model.depth_of_burial_m == 3.0
    assert model.b_scaled_burial > 0.0


def test_noise_model_invalid_params():
    with pytest.raises(ValueError):
        NoiseModel(charge_per_delay_kg=-10.0, depth_of_burial_m=3.0)
    with pytest.raises(ValueError):
        NoiseModel(charge_per_delay_kg=150.0, depth_of_burial_m=-1.0)


def test_noise_model_calculate_peak_overpressure():
    model = NoiseModel(charge_per_delay_kg=150.0, depth_of_burial_m=3.0)
    res = model.calculate_peak_overpressure(distance_m=500.0)
    assert "overpressure_kpa" in res
    assert "spl_db" in res
    assert res["distance_m"] == 500.0
    assert res["overpressure_kpa"] > 0.0
    assert res["spl_db"] > 0.0


def test_noise_model_calculate_noise_at_distance():
    model = NoiseModel(charge_per_delay_kg=200.0, depth_of_burial_m=2.0)
    res = model.calculate_noise_at_distance(distance_m=300.0)
    assert "spl_db" in res
    assert "exceeds_limit" in res
    assert isinstance(res["exceeds_limit"], bool)


def test_noise_model_calculate_zone_of_impact():
    model = NoiseModel(charge_per_delay_kg=150.0, depth_of_burial_m=3.0)
    res = model.calculate_zone_of_impact(limit_db=120.0)
    assert "distance_to_limit_m" in res
    assert res["limit_db"] == 120.0
    assert res["distance_to_limit_m"] > 0.0


def test_noise_model_calculate_equivalent_continuous_level():
    model = NoiseModel(charge_per_delay_kg=100.0, depth_of_burial_m=2.0)
    events = [
        {"spl_db": 115.0, "duration_s": 2.0},
        {"spl_db": 110.0, "duration_s": 2.0},
    ]
    leq = model.calculate_equivalent_continuous_level(events=events, duration_s=3600.0)
    assert leq > 0.0


def test_predict_noise_overpressure_legacy():
    result = predict_noise_overpressure(max_charge_per_delay_kg=150.0, distance_m=500.0)
    assert "noise_dbl" in result
    assert "overpressure_kpa" in result
    assert "scaled_distance" in result
    assert "exceeds_botswana_limit" in result
