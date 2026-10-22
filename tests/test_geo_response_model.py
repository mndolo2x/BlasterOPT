"""
Unit tests for GeoResponseModel wrapper class (models/custom/geo_response_model.py).
"""

import numpy as np
import pandas as pd
import pytest

from models.base import BaseBlastModel, ModelMetadata
from models.custom.geo_response_model import GeoResponseModel


def test_geo_response_model_metadata():
    """Verify GeoResponseModel metadata schema and attributes."""
    meta = GeoResponseModel.get_metadata()
    assert isinstance(meta, ModelMetadata)
    assert meta.name == "geo_response"
    assert "GeoResponse" in meta.display_name
    assert meta.task_type == "classification"
    assert meta.output_features == ["rock_class"]
    assert len(meta.input_features) == 8


def test_geo_response_model_fit_predict():
    """Verify fit, predict, and predict_proba execution on GeoResponseModel."""
    model = GeoResponseModel()
    assert isinstance(model, BaseBlastModel)
    assert model.is_fitted is False

    df_sample = pd.DataFrame({
        "penetration_rate_m_hr": [20.0, 35.0, 45.0, 15.0, 50.0],
        "percussion_pressure_bar": [120.0, 150.0, 180.0, 110.0, 200.0],
        "feed_pressure_bar": [80.0, 90.0, 100.0, 85.0, 110.0],
        "rotation_pressure_bar": [80.0, 95.0, 110.0, 85.0, 120.0],
        "flushing_pressure_bar": [8.0, 10.0, 12.0, 7.0, 15.0],
        "rotation_speed_rpm": [150, 180, 200, 160, 220],
        "depth_m": [3.0, 3.5, 4.0, 2.5, 5.0],
        "feed_percussion_ratio": [0.4, 0.5, 0.6, 0.45, 0.65],
        "rock_class": ["Soft", "Medium", "Hard", "Soft", "Very Hard"]
    })

    X = df_sample.drop(columns=["rock_class"])
    y = df_sample["rock_class"]

    model.fit(X, y)
    assert model.is_fitted is True

    preds_df = model.predict(X)
    assert "rock_class" in preds_df.columns
    assert "confidence_score" in preds_df.columns
    assert "review_flag" in preds_df.columns
    assert len(preds_df) == len(X)

    probas = model.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape[0] == len(X)
