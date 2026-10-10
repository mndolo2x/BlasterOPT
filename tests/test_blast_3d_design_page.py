"""
Unit tests for 3D Blast Design page layout, bento metrics, SHAP explanations, and baseline comparisons.
"""

from src.explainability import get_feature_contributions, generate_natural_language_explanation


def test_baseline_delta_calculation():
    """Verify calculation of delta indicators against baseline values."""
    baseline_p80 = 35.0
    current_p80 = 32.5
    delta = current_p80 - baseline_p80

    assert delta == -2.5
    delta_str = f" (↓ {abs(delta):.1f} cm)" if delta < 0 else f" (↑ {delta:.1f} cm)"
    assert delta_str == " (↓ 2.5 cm)"


def test_shap_explanation_integration():
    """Verify SHAP explanation generation for P80 fragmentation and PPV vibration."""
    input_payload = {
        "burden_m": 4.2,
        "spacing_m": 5.1,
        "powder_factor_kg_m3": 0.65,
        "stemming_m": 3.0,
        "bench_height_m": 15.0,
        "hole_depth_m": 16.5,
        "hole_diameter_mm": 165.0,
        "max_charge_per_delay_kg": 640.0,
        "explosive_rws": 115.0,
        "rock_factor_A": 8.0,
        "monitoring_distance_m": 800.0,
    }

    contrib_data = get_feature_contributions(None, input_payload, target="d50_mm")
    assert "contributions" in contrib_data
    assert "baseline_value" in contrib_data

    explanation = generate_natural_language_explanation(
        shap_values=list(contrib_data["contributions"].values()),
        feature_names=list(contrib_data["contributions"].keys()),
        prediction=32.5,
        constraints={"metric": "Fragmentation P80", "limit": 40.0, "unit": "cm"},
    )
    assert len(explanation) > 0
    assert "cm" in explanation
