import pytest
from src.predict import predict_outcomes, predict_physics_fallback, predict_single_blast


def test_predict_outcomes_five_targets():
    """Test that predict_outcomes returns a dictionary with keys for all five output targets."""
    valid_inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 7.0,
        "spacing_m": 8.0,
        "stemming_m": 5.0,
        "charge_mass_per_hole_kg": 350.0,
        "powder_factor_kg_m3": 0.65,
        "max_charge_per_delay_kg": 700.0,
        "monitoring_distance_m": 450.0,
        "explosive_rws": 100.0,
    }

    result = predict_outcomes(valid_inputs)
    assert isinstance(result, dict)
    assert "error" not in result

    expected_targets = ["d50_mm", "uniformity_index_n", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]
    for target in expected_targets:
        assert target in result, f"Target {target} missing from prediction output"
        assert isinstance(result[target], (int, float))
        assert result[target] > 0, f"Target {target} should be positive"


def test_predict_outcomes_missing_inputs():
    """Test that predict_outcomes handles missing input parameters gracefully."""
    incomplete_inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        # missing hole_diameter_mm, burden_m, etc.
    }

    result = predict_outcomes(incomplete_inputs)
    assert isinstance(result, dict)
    assert "error" in result
    assert "Missing required input parameter" in result["error"]


def test_predict_outcomes_out_of_range_inputs():
    """Test that predict_outcomes handles out-of-range inputs gracefully."""
    out_of_range_inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": -5.0,  # invalid bench height
        "hole_diameter_mm": 250.0,
        "burden_m": 7.0,
        "spacing_m": 8.0,
        "stemming_m": 5.0,
        "charge_mass_per_hole_kg": 350.0,
        "powder_factor_kg_m3": 0.65,
        "max_charge_per_delay_kg": 700.0,
        "monitoring_distance_m": 450.0,
    }

    result = predict_outcomes(out_of_range_inputs)
    assert isinstance(result, dict)
    assert "error" in result
    assert "Out-of-range input parameter" in result["error"]


def test_predict_outcomes_invalid_type_input():
    """Test that predict_outcomes handles non-dictionary inputs gracefully."""
    result = predict_outcomes("invalid input string")
    assert isinstance(result, dict)
    assert "error" in result
