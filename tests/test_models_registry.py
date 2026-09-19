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
    register_model,
    RandomForestBlastModel,
    XGBoostBlastModel,
    RidgeBlastModel,
    GAANNBlastModel,
    PINNBlastModel,
    EnsembleBlastModel,
    SiteCalibrationBlastModel,
)


def test_model_metadata_schemas():
    """
    Test ModelMetadata Pydantic model across custom and sklearn models.
    """
    rf_meta = RandomForestBlastModel.get_metadata()
    pinn_meta = PINNBlastModel.get_metadata()
    ens_meta = EnsembleBlastModel.get_metadata()

    assert rf_meta.name == "random_forest"
    assert rf_meta.model_type == "sklearn"

    assert pinn_meta.name == "pinn"
    assert pinn_meta.model_type == "physics_informed"
    assert pinn_meta.supports_uncertainty is True

    assert ens_meta.name == "ensemble"
    assert ens_meta.supports_uncertainty is True


def test_streamlit_dropdown_auto_discovery():
    """
    Test get_available_models() returns all custom and sklearn models for Streamlit UI dropdown.
    """
    available = ModelRegistry.get_available_models()
    keys = [k for display_name, k in available]

    assert "ga_ann" in keys
    assert "pinn" in keys
    assert "ensemble" in keys
    assert "site_calibration" in keys
    assert "random_forest" in keys
    assert "xgboost" in keys
    assert "ridge" in keys

    meta_map = ModelRegistry.get_model_metadata_map()
    assert "pinn" in meta_map
    assert meta_map["pinn"].display_name == "Physics-Informed Neural Network (PINN)"


def test_pandas_dataframe_predict_and_fit():
    """
    Test model predict with pandas DataFrames.
    """
    X_df = pd.DataFrame(np.random.uniform(1.0, 10.0, size=(10, 12)), columns=[
        "burden_m", "spacing_m", "hole_diameter_mm", "bench_height_m",
        "stemming_m", "subdrill_m", "powder_factor_kg_m3", "max_charge_per_delay_kg",
        "rock_factor_a", "rmr", "monitoring_distance_m", "explosive_rws"
    ])
    y_df = pd.DataFrame(np.random.uniform(10.0, 200.0, size=(10, 4)), columns=[
        "d50_mm", "ppv_mms", "flyrock_m", "cost_usd"
    ])

    rf_model = ModelRegistry.get_model("random_forest")
    rf_model.fit(X_df, y_df)

    preds = rf_model.predict(X_df)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (10, 4)
    assert "d50_mm" in preds.columns


def test_uncertainty_and_explain_hooks():
    """
    Test predict_with_uncertainty and explain hooks on custom models.
    """
    X = np.random.uniform(1.0, 10.0, size=(10, 12))
    Y = np.random.uniform(10.0, 200.0, size=(10, 4))

    pinn_model = ModelRegistry.get_model("pinn")
    pinn_model.fit(X, Y)

    unc_res = pinn_model.predict_with_uncertainty(X)
    assert unc_res is not None
    assert "mean" in unc_res
    assert "std" in unc_res

    rf_model = ModelRegistry.get_model("random_forest")
    rf_model.fit(X, Y)
    exp_res = rf_model.explain(X)
    assert exp_res is not None
    assert "feature_importances" in exp_res
