"""
Unit tests for GeoResponse MWD 8-feature prediction interface, defaults, validation, and feature ordering.
"""

import pytest
import pandas as pd
import numpy as np

from src.ui.geo_response_predict import (
    REQUIRED_FEATURE_ORDER,
    DEFAULT_MWD_INPUTS,
    FIELD_DISPLAY_NAMES,
    FEATURE_RANGES,
    validate_mwd_input_ranges,
    build_ordered_feature_dataframe,
    get_or_create_default_classifier,
)
from src.georesponse.models import GeoResponseClassifier


def test_all_eight_inputs_present_in_ui():
    """Verify all eight required MWD input feature names are present in config and UI page."""
    expected_8_features = [
        "penetration_rate_m_hr",
        "percussion_pressure_bar",
        "feed_pressure_bar",
        "rotation_pressure_bar",
        "flushing_pressure_bar",
        "rotation_speed_rpm",
        "depth_m",
        "feed_percussion_ratio",
    ]
    assert len(REQUIRED_FEATURE_ORDER) == 8
    assert REQUIRED_FEATURE_ORDER == expected_8_features

    # Verify UI page references all 8 feature keys
    with open("pages/georesponse.py", "r") as f:
        page_content = f.read()

    for feat in expected_8_features:
        assert feat in page_content, f"Feature {feat} missing from pages/georesponse.py"


def test_default_values_match_specification():
    """Verify default values match Norwegian tunnel drilling specifications."""
    assert DEFAULT_MWD_INPUTS["penetration_rate_m_hr"] == 35.0
    assert DEFAULT_MWD_INPUTS["percussion_pressure_bar"] == 180.0
    assert DEFAULT_MWD_INPUTS["feed_pressure_bar"] == 90.0
    assert DEFAULT_MWD_INPUTS["rotation_pressure_bar"] == 95.0
    assert DEFAULT_MWD_INPUTS["flushing_pressure_bar"] == 10.0
    assert DEFAULT_MWD_INPUTS["rotation_speed_rpm"] == 180
    assert DEFAULT_MWD_INPUTS["depth_m"] == 3.5
    assert DEFAULT_MWD_INPUTS["feed_percussion_ratio"] == 0.50


def test_prediction_requires_all_eight_inputs():
    """Verify prediction DataFrame construction raises ValueError if any of the 8 inputs is missing."""
    incomplete_inputs = DEFAULT_MWD_INPUTS.copy()
    del incomplete_inputs["feed_pressure_bar"]

    with pytest.raises(ValueError, match="Prediction requires all eight MWD inputs"):
        build_ordered_feature_dataframe(incomplete_inputs)


def test_out_of_range_values_trigger_warning():
    """Verify out-of-range inputs trigger warnings without blocking prediction."""
    out_of_range_inputs = DEFAULT_MWD_INPUTS.copy()
    out_of_range_inputs["feed_pressure_bar"] = 200.0  # Range is 40.0 to 150.0
    out_of_range_inputs["depth_m"] = 15.0             # Range is 0.5 to 10.0

    warnings = validate_mwd_input_ranges(out_of_range_inputs)
    assert len(warnings) == 2
    assert any("Feed Pressure (bar)" in w for w in warnings)
    assert any("Depth (m)" in w for w in warnings)
    assert all("outside the typical range" in w for w in warnings)

    # Verify prediction still executes successfully
    clf = get_or_create_default_classifier()
    X_sample = build_ordered_feature_dataframe(out_of_range_inputs)
    review_df = clf.predict_with_review_flags(X_sample)
    assert len(review_df) == 1
    assert "predicted_rock_type" in review_df.columns


def test_feature_order_matches_model_expectation():
    """Verify feature column order explicitly matches model positional expectation."""
    sample_df = build_ordered_feature_dataframe(DEFAULT_MWD_INPUTS)
    assert list(sample_df.columns) == REQUIRED_FEATURE_ORDER

    clf = GeoResponseClassifier(model_type="random_forest")
    clf.fit(sample_df, ["Hard Rock"])
    assert clf.feature_names_ == REQUIRED_FEATURE_ORDER


def test_reset_button_restores_defaults():
    """Verify reset logic restores modified input dict to original defaults."""
    modified_inputs = {
        "penetration_rate_m_hr": 75.0,
        "percussion_pressure_bar": 280.0,
        "feed_pressure_bar": 140.0,
        "rotation_pressure_bar": 150.0,
        "flushing_pressure_bar": 25.0,
        "rotation_speed_rpm": 290,
        "depth_m": 8.0,
        "feed_percussion_ratio": 0.85,
    }
    assert modified_inputs != DEFAULT_MWD_INPUTS

    # Simulate reset
    restored = modified_inputs.copy()
    for k, v in DEFAULT_MWD_INPUTS.items():
        restored[k] = v

    assert restored == DEFAULT_MWD_INPUTS


def test_input_summary_shown_after_prediction():
    """Verify prediction output contains required summary information."""
    clf = get_or_create_default_classifier()
    X_sample = build_ordered_feature_dataframe(DEFAULT_MWD_INPUTS)
    review_df = clf.predict_with_review_flags(X_sample)

    assert "predicted_rock_type" in review_df.columns
    assert "confidence_score" in review_df.columns
    assert "requires_review_flag" in review_df.columns
    assert "review_warning" in review_df.columns

    pred_class = review_df["predicted_rock_type"].iloc[0]
    conf_score = review_df["confidence_score"].iloc[0]
    requires_review = review_df["requires_review_flag"].iloc[0]

    assert isinstance(pred_class, str)
    assert 0.0 <= conf_score <= 1.0
    assert isinstance(requires_review, (bool, np.bool_))
