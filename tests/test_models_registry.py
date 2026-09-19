"""
Unit tests for Model Registry Architecture (`models/registry.py`).
"""

import os
import pytest
import numpy as np
import pandas as pd
from models import (
    BaseBlastModel,
    ModelMetadata,
    ModelRegistry,
    get_registry,
    RandomForestModel,
    RandomForestBlastModel,
    XGBoostBlastModel,
    RidgeBlastModel,
    GAANNBlastModel,
    PINNBlastModel,
    EnsembleBlastModel,
    SiteCalibrationBlastModel,
)


def test_get_registry_singleton_and_discover():
    """
    Test get_registry() singleton and discover() scanning models/ directory.
    """
    registry = get_registry()
    assert registry is not None

    models_meta = registry.list_models()
    names = [meta.name for meta in models_meta]

    assert "random_forest" in names
    assert "xgboost" in names
    assert "ridge" in names
    assert "ga_ann" in names
    assert "pinn" in names
    assert "ensemble" in names
    assert "site_calibration" in names


def test_random_forest_model_metadata_and_dataframe_predict(tmp_path):
    """
    Test RandomForestModel metadata, fit/predict with DataFrame, and joblib save/load.
    """
    meta = RandomForestModel.get_metadata()
    assert meta.name == "random_forest"
    assert meta.display_name == "Random Forest"
    assert meta.output_features == ["fragmentation_p80", "ppv", "airblast"]

    X_df = pd.DataFrame({
        "burden": [3.5, 4.0, 3.8],
        "spacing": [4.5, 5.0, 4.8],
        "powder_factor": [0.6, 0.8, 0.7],
        "stemming": [3.0, 3.5, 3.2],
        "rock_factor": [8.0, 9.0, 8.5]
    }, index=[10, 20, 30])

    y_df = pd.DataFrame({
        "fragmentation_p80": [220.0, 180.0, 200.0],
        "ppv": [5.2, 8.1, 6.5],
        "airblast": [115.0, 122.0, 118.0]
    }, index=[10, 20, 30])

    rf = RandomForestModel(n_estimators=10, random_state=42)
    rf.fit(X_df, y_df)

    preds = rf.predict(X_df)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == ["fragmentation_p80", "ppv", "airblast"]
    assert list(preds.index) == [10, 20, 30]

    # Save and load
    save_file = str(tmp_path / "rf_model.joblib")
    rf.save(save_file)

    loaded_rf = RandomForestModel.load(save_file)
    loaded_preds = loaded_rf.predict(X_df)
    assert isinstance(loaded_preds, pd.DataFrame)
    assert loaded_preds.shape == (3, 3)


def test_registry_get_model_and_metadata():
    """
    Test registry.get_model() and registry.get_metadata().
    """
    registry = get_registry()

    pinn_cls = registry.get_model("pinn")
    pinn_meta = registry.get_metadata("pinn")

    assert pinn_cls == PINNBlastModel
    assert pinn_meta.display_name == "Physics-Informed Neural Network (PINN)"
    assert pinn_meta.supports_uncertainty is True


def test_get_available_models_streamlit_dropdown():
    """
    Test get_available_models() returns tuples for Streamlit dropdown.
    """
    registry = get_registry()
    available = registry.get_available_models()

    keys = [key for disp, key in available]

    assert "ga_ann" in keys
    assert "pinn" in keys
    assert "ensemble" in keys
    assert "site_calibration" in keys


def test_registry_load_errors_list():
    """
    Test get_load_errors() returns error tracking list.
    """
    registry = get_registry()
    errors = registry.get_load_errors()
    assert isinstance(errors, list)
