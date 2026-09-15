import pytest
import pandas as pd
from src.predict import predict_outcomes, predict_physics_fallback, predict_single_blast, predict_crusher_throughput, total_cost_per_tonne, find_similar_blasts
from src.synthetic_data import generate_synthetic_blast_data


def test_find_similar_blasts():
    """Test find_similar_blasts function returning top_k similar records."""
    hist_df = generate_synthetic_blast_data(num_samples=50, seed=42)
    sample_blast = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "charge_mass_per_hole_kg": 320.0,
        "powder_factor_kg_m3": 0.65,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }

    similar_df = find_similar_blasts(sample_blast, hist_df, top_k=5)
    assert isinstance(similar_df, pd.DataFrame)
    assert len(similar_df) == 5
    assert "similarity_distance" in similar_df.columns


def test_total_cost_per_tonne():
    """Test total_cost_per_tonne function return components and positive total cost."""
    blast_params = {
        "powder_factor_kg_m3": 0.65,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "d50_mm": 220.0,
    }
    cost_dict = total_cost_per_tonne(blast_params)
    assert isinstance(cost_dict, dict)
    assert "total_cost_usd_t" in cost_dict
    assert cost_dict["total_cost_usd_t"] > 0
    assert cost_dict["drilling_cost_usd_t"] > 0
    assert cost_dict["explosive_cost_usd_t"] > 0
    assert cost_dict["digging_cost_usd_t"] > 0
    assert cost_dict["hauling_cost_usd_t"] > 0
    assert cost_dict["crushing_cost_usd_t"] > 0
    assert cost_dict["milling_cost_usd_t"] > 0


def test_predict_crusher_throughput():
    """Test predict_crusher_throughput function return structure and positive values."""
    res = predict_crusher_throughput(d80_cm=25.0, ore_hardness=14.0, crusher_settings={"css_mm": 150.0, "power_rating_kw": 400.0})
    assert isinstance(res, dict)
    assert "throughput_tph" in res
    assert "specific_energy_kwh_t" in res
    assert res["throughput_tph"] > 0
    assert res["specific_energy_kwh_t"] > 0


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
