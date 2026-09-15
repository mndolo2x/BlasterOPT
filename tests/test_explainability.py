import pytest
import pytest
from src.explainability import get_feature_contributions, plot_feature_contributions_waterfall, explain_prediction, HAS_SHAP
from src.models import HAS_TORCH

if HAS_TORCH:
    import torch
    from src.models import GAANNModel


def test_get_feature_contributions_structure():
    """Test that get_feature_contributions returns expected structure and non-empty contributions."""
    sample_inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }

    res = get_feature_contributions(model_pipeline=None, input_payload=sample_inputs, target="d50_mm")
    assert isinstance(res, dict)
    assert "target" in res
    assert "baseline_value" in res
    assert "predicted_value" in res
    assert "contributions" in res
    assert isinstance(res["contributions"], dict)
    assert len(res["contributions"]) > 0


def test_get_feature_contributions_invalid_input():
    """Test that get_feature_contributions handles invalid input payloads gracefully."""
    res = get_feature_contributions(model_pipeline=None, input_payload="invalid string")
    assert isinstance(res, dict)
    assert "error" in res


def test_explain_prediction_shap():
    """Test explain_prediction function when shap and torch are available or missing."""
    if not HAS_SHAP or not HAS_TORCH:
        with pytest.raises(ImportError):
            explain_prediction(None, None)
    else:
        model = GAANNModel(input_size=10)
        x = torch.randn(1, 10)
        bg = torch.zeros(10, 10)
        shap_vals = explain_prediction(model, x, background_data=bg)
        assert shap_vals is not None


def test_plot_feature_contributions_waterfall():
    """Test that plot_feature_contributions_waterfall returns a Plotly Figure."""
    sample_inputs = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }

    res = get_feature_contributions(model_pipeline=None, input_payload=sample_inputs, target="d50_mm")
    fig = plot_feature_contributions_waterfall(res)
    assert fig is not None
    assert hasattr(fig, "data")
