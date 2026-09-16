import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.explainability import (
    get_feature_contributions,
    plot_feature_contributions_waterfall,
    explain_prediction,
    get_shap_explanation,
    get_lime_explanation,
    generate_natural_language_explanation,
    create_explanation_panel,
    HAS_SHAP,
    HAS_LIME,
)
from src.models import HAS_TORCH

if HAS_TORCH:
    import torch
    from src.models import GAANNModel, ANN_RF_Ensemble


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
        res = explain_prediction(None, np.zeros((1, 5)))
        assert isinstance(res, dict)
    else:
        model = GAANNModel(input_size=10)
        x = torch.randn(1, 10)
        bg = torch.zeros(10, 10)
        res = explain_prediction(model, x, background_data=bg)
        assert isinstance(res, dict)
        assert "shap_values" in res


def test_get_shap_explanation_keys():
    """Test get_shap_explanation returns a dict with the correct required keys."""
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X, y)

    input_df = pd.DataFrame(X[:1], columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    res = get_shap_explanation(rf, input_df)

    assert isinstance(res, dict)
    expected_keys = [
        "shap_values",
        "base_value",
        "feature_names",
        "force_plot",
        "waterfall_plot",
        "feature_contributions",
    ]
    for key in expected_keys:
        assert key in res, f"Expected key '{key}' in get_shap_explanation result"
    assert len(res["feature_contributions"]) == 4


def test_get_shap_explanation_rf():
    """Test get_shap_explanation with a Random Forest model."""
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X, y)

    input_df = pd.DataFrame(X[:1], columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    res = get_shap_explanation(rf, input_df)

    assert isinstance(res, dict)
    assert "shap_values" in res
    assert "base_value" in res
    assert "force_plot" in res
    assert "waterfall_plot" in res
    assert "feature_contributions" in res
    assert len(res["feature_contributions"]) == 4


def test_get_lime_explanation_rf():
    """Test get_lime_explanation with a Random Forest model."""
    X = np.random.randn(30, 4)
    y = np.random.randn(30)
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X, y)

    feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
    input_df = pd.DataFrame(X[:1], columns=feature_names)
    train_df = pd.DataFrame(X, columns=feature_names)

    res = get_lime_explanation(rf, input_df, training_data=train_df, feature_names=feature_names)

    assert isinstance(res, dict)
    assert "lime_explanation" in res
    assert "feature_weights" in res
    assert len(res["feature_weights"]) > 0


def test_get_lime_explanation_torch():
    """Test get_lime_explanation with a PyTorch ANN model."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    model = GAANNModel(input_size=5)
    X = np.random.randn(20, 5)
    feature_names = [f"f_{i}" for i in range(5)]

    res = get_lime_explanation(
        model,
        X[:1],
        training_data=X,
        feature_names=feature_names,
        model_type="ann",
    )

    assert isinstance(res, dict)
    assert "lime_explanation" in res
    assert "feature_weights" in res


def test_generate_natural_language_explanation():
    """Test generate_natural_language_explanation plain English text generation and word count."""
    shap_vals = [0.12, 0.08, 0.05, -0.05, -0.02]
    feature_names = ["spacing_m", "rock_factor_A", "powder_factor_kg_m3", "stemming_m", "burden_m"]
    prediction = 5.8
    constraints = {"metric": "vibration", "limit": 5.0, "unit": "mm/s"}

    explanation = generate_natural_language_explanation(shap_vals, feature_names, prediction, constraints)

    assert isinstance(explanation, str)
    assert len(explanation) > 0, "Explanation string should be non-empty"
    assert "Vibration is predicted to exceed the limit" in explanation
    assert "5.0 mm/s" in explanation
    assert "5.8" in explanation
    assert len(explanation.split()) <= 150


def test_generate_natural_language_explanation_features_and_direction():
    """Test that generate_natural_language_explanation contains at least one feature name and one direction word."""
    shap_vals = [0.15, -0.10]
    feature_names = ["powder_factor_kg_m3", "stemming_m"]
    prediction = 8.5
    constraints = {"metric": "PPV", "limit": 10.0, "unit": "mm/s"}

    explanation = generate_natural_language_explanation(shap_vals, feature_names, prediction, constraints)

    assert isinstance(explanation, str)
    assert len(explanation.strip()) > 0, "Expected non-empty string"

    # Check for feature names (either raw or formatted)
    has_feature_name = any(
        fn in explanation or fn.replace("_", " ") in explanation or "powder factor" in explanation or "stemming length" in explanation
        for fn in feature_names
    )
    assert has_feature_name, f"Expected at least one feature name in explanation: {explanation}"

    # Check for direction words ("increase", "decrease", "reduced", "increased", "increasing", "reducing")
    direction_words = ["increase", "decrease", "reduced", "increased", "increasing", "reducing"]
    has_direction_word = any(dw in explanation.lower() for dw in direction_words)
    assert has_direction_word, f"Expected at least one direction word in explanation: {explanation}"


def test_create_explanation_panel():
    """Test create_explanation_panel consolidated dictionary function."""
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X, y)

    feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
    input_df = pd.DataFrame(X[:1], columns=feature_names)
    train_df = pd.DataFrame(X, columns=feature_names)

    panel = create_explanation_panel(
        model=rf,
        input_data=input_df,
        feature_names=feature_names,
        training_data=train_df,
        prediction=12.5,
        constraints={"metric": "d50 fragmentation", "limit": 15.0, "unit": "cm"},
    )

    assert isinstance(panel, dict)
    assert "shap" in panel
    assert "lime" in panel
    assert "natural_language" in panel
    assert isinstance(panel["natural_language"], str)


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
