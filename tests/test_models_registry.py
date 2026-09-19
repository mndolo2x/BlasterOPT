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
    PINNModel,
    PINNBlastModel,
    EnsembleModel,
    EnsembleBlastModel,
    SiteCalibrationModel,
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


def test_supports_pipeline_training_metadata_filter():
    """
    Test filtering models by supports_pipeline_training flag.
    """
    rf_meta = RandomForestModel.get_metadata()
    pinn_meta = PINNModel.get_metadata()

    assert rf_meta.supports_pipeline_training is True
    assert pinn_meta.supports_pipeline_training is False

    trainable = [m for m in get_registry().list_models() if m.supports_pipeline_training]
    trainable_names = [m.name for m in trainable]

    assert "random_forest" in trainable_names
    assert "xgboost" in trainable_names
    assert "ridge" in trainable_names
    assert "pinn" not in trainable_names


def test_custom_models_metadata_predict_and_save_load(tmp_path):
    """
    Test PINNModel, EnsembleModel, and SiteCalibrationModel metadata, predict, and save/load.
    """
    # 1. PINNModel
    pinn_meta = PINNModel.get_metadata()
    assert pinn_meta.name == "pinn"
    assert pinn_meta.output_features == ["fragmentation_p80", "ppv", "airblast"]

    X_df = pd.DataFrame({
        "burden": [3.5, 4.0],
        "spacing": [4.5, 5.0],
        "powder_factor": [0.6, 0.8],
        "stemming": [3.0, 3.5],
        "rock_factor": [8.0, 9.0],
        "blastability_index": [60.0, 65.0],
        "charge_per_delay": [300.0, 350.0]
    }, index=[101, 102])

    y_df = pd.DataFrame({
        "fragmentation_p80": [220.0, 180.0],
        "ppv": [5.2, 8.1],
        "airblast": [115.0, 122.0]
    }, index=[101, 102])

    pinn = PINNModel()
    pinn.fit(X_df, y_df)
    preds = pinn.predict(X_df)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == ["fragmentation_p80", "ppv", "airblast"]
    assert list(preds.index) == [101, 102]

    pinn_path = str(tmp_path / "pinn.joblib")
    pinn.save(pinn_path)
    loaded_pinn = PINNModel.load(pinn_path)
    assert loaded_pinn.predict(X_df).shape == (2, 3)

    # 2. EnsembleModel
    ens_meta = EnsembleModel.get_metadata()
    assert ens_meta.name == "ensemble"
    assert ens_meta.supports_uncertainty is True

    ens = EnsembleModel()
    ens.fit(X_df, np.random.rand(2, 4))
    ens_preds = ens.predict(X_df)
    assert isinstance(ens_preds, pd.DataFrame)
    assert list(ens_preds.columns) == ["fragmentation_p80", "ppv", "airblast"]

    unc_res = ens.predict_with_uncertainty(X_df)
    assert "mean" in unc_res
    assert isinstance(unc_res["mean"], pd.DataFrame)

    ens_path = str(tmp_path / "ens.joblib")
    ens.save(ens_path)
    loaded_ens = EnsembleModel.load(ens_path)
    assert loaded_ens.predict(X_df).shape == (2, 3)

    # 3. SiteCalibrationModel
    cal_meta = SiteCalibrationModel.get_metadata()
    assert cal_meta.name == "site_calibration"

    cal = SiteCalibrationModel()
    cal.fit(X_df, y_df)
    cal_preds = cal.predict(X_df)
    assert isinstance(cal_preds, pd.DataFrame)
    assert list(cal_preds.columns) == ["fragmentation_p80", "ppv", "airblast"]

    cal_path = str(tmp_path / "cal.joblib")
    cal.save(cal_path)
    loaded_cal = SiteCalibrationModel.load(cal_path)
    assert loaded_cal.predict(X_df).shape == (2, 3)
