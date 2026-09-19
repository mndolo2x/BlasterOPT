"""
Unit tests for Model Registry Architecture (`models/registry.py`).
"""

import pytest
import numpy as np
import pandas as pd
from models import (
    BaseBlastModel,
    ModelMetadata,
    ModelRegistry,
    get_registry,
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

    display_names = [disp for disp, key in available]
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
