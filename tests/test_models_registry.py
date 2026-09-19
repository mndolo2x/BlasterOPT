"""
Unit tests for Model Registry Architecture (`models/registry.py`).
"""

import pytest
import numpy as np
from models import (
    BaseBlastModel,
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


def test_model_registry_list_and_get():
    """
    Test ModelRegistry auto-discovery and model retrieval.
    """
    model_list = ModelRegistry.list_models()
    assert "random_forest" in model_list
    assert "xgboost" in model_list
    assert "ridge" in model_list
    assert "ga_ann" in model_list
    assert "pinn" in model_list
    assert "ensemble" in model_list
    assert "site_calibration" in model_list

    rf_cls = ModelRegistry.get_model_class("random_forest")
    assert rf_cls == RandomForestBlastModel


def test_model_instantiation_and_fit_predict():
    """
    Test instantiating and training models retrieved from registry.
    """
    X = np.random.uniform(1.0, 10.0, size=(20, 12))
    Y = np.random.uniform(10.0, 200.0, size=(20, 4))

    # Instantiate via registry
    rf_model = ModelRegistry.get_model("random_forest")
    rf_model.fit(X, Y)
    preds = rf_model.predict(X)

    assert preds.shape == (20, 4)
    assert np.all(preds > 0.0)

    eval_res = rf_model.evaluate(X, Y)
    assert "mean_r2" in eval_res
    assert "mean_rmse" in eval_res


def test_custom_model_registration_decorator():
    """
    Test @register_model decorator for custom user model.
    """
    @register_model("dummy_test_model")
    class DummyBlastModel(BaseBlastModel):
        def fit(self, X, Y):
            self.is_fitted = True
            return self
        def predict(self, X):
            return np.tile([100.0, 5.0, 50.0, 2.0], (len(X), 1))

    model = ModelRegistry.get_model("dummy_test_model")
    assert isinstance(model, DummyBlastModel)

    X_dummy = np.random.rand(5, 12)
    preds = model.predict(X_dummy)
    assert preds.shape == (5, 4)


def test_yaml_config_loading():
    """
    Test YAML config loading in ModelRegistry.
    """
    cfg = ModelRegistry.load_registry_config()
    assert "random_forest" in cfg
    assert "pinn" in cfg
    assert cfg["pinn"]["type"] == "custom"
