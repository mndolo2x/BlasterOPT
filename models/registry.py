"""
Model Registry Submodule (`models/registry.py`).
Implements auto-discovery, dynamic registration decorator, model instantiation,
and Streamlit UI dropdown metadata mapping.
"""

import os
import yaml
import importlib
import logging
from typing import Dict, Any, Type, Optional, List, Tuple
from models.base import BaseBlastModel, ModelMetadata

logger = logging.getLogger(__name__)


_REGISTERED_MODELS: Dict[str, Type[BaseBlastModel]] = {}


def register_model(name: str):
    """
    Decorator to register model class into global ModelRegistry.
    """
    def decorator(cls: Type[BaseBlastModel]):
        if not issubclass(cls, BaseBlastModel):
            raise TypeError(f"Registered class {cls.__name__} must inherit from BaseBlastModel")
        _REGISTERED_MODELS[name.lower()] = cls
        logger.info(f"Registered model '{name}' -> {cls.__name__}")
        return cls
    return decorator


class ModelRegistry:
    """
    Registry for managing, instantiating, and discovering blast models for Streamlit ML Manager.
    """

    _config_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def load_registry_config(cls, yaml_path: str = "models/configs/model_registry.yaml") -> Dict[str, Any]:
        """Loads model registry metadata configuration from YAML file."""
        if cls._config_cache is None and os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                    cls._config_cache = data.get("models", {}) if data else {}
            except Exception as e:
                logger.error(f"Error loading model registry config from {yaml_path}: {e}")
                cls._config_cache = {}
        return cls._config_cache or {}

    @classmethod
    def auto_discover(cls):
        """
        Auto-discovers and imports all model modules under models/ to trigger registration.
        Handles import errors gracefully.
        """
        modules_to_import = [
            "models.sklearn_wrappers.random_forest",
            "models.sklearn_wrappers.xgboost",
            "models.sklearn_wrappers.ridge",
            "models.custom.ga_ann",
            "models.custom.pinn",
            "models.custom.ensemble",
            "models.custom.site_calibration_model",
        ]

        for mod_name in modules_to_import:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                logger.warning(f"Auto-discovery skipped broken/missing model module '{mod_name}': {e}")

    @classmethod
    def list_models(cls) -> List[str]:
        """Lists internal keys of all registered models."""
        cls.auto_discover()
        return list(_REGISTERED_MODELS.keys())

    @classmethod
    def get_available_models(cls) -> List[Tuple[str, str]]:
        """
        Returns list of (display_name, model_key) tuples for Streamlit dropdown selection.
        All models (GA-ANN, PINN, Ensemble, Site Calibration, RandomForest, XGBoost, Ridge) appear automatically.
        """
        cls.auto_discover()
        models_info = []
        for key, cls_type in _REGISTERED_MODELS.items():
            try:
                meta = cls_type.get_metadata()
                models_info.append((meta.display_name, key))
            except Exception:
                models_info.append((key.replace("_", " ").title(), key))
        return sorted(models_info, key=lambda x: x[0])

    @classmethod
    def get_model_metadata_map(cls) -> Dict[str, ModelMetadata]:
        """
        Returns map of model_key -> ModelMetadata for Streamlit UI display.
        """
        cls.auto_discover()
        meta_map = {}
        for key, cls_type in _REGISTERED_MODELS.items():
            try:
                meta_map[key] = cls_type.get_metadata()
            except Exception as e:
                logger.warning(f"Could not retrieve metadata for {key}: {e}")
        return meta_map

    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseBlastModel]:
        """Retrieves model class by registered name."""
        cls.auto_discover()
        key = name.lower()
        if key not in _REGISTERED_MODELS:
            raise KeyError(f"Model '{name}' is not registered. Available: {cls.list_models()}")
        return _REGISTERED_MODELS[key]

    @classmethod
    def get_model(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseBlastModel:
        """
        Instantiates registered model by name with parameters from config or YAML.
        """
        model_cls = cls.get_model_class(name)
        cfg = config or {}

        # Merge YAML params if available
        yaml_configs = cls.load_registry_config()
        if name.lower() in yaml_configs:
            yaml_params = yaml_configs[name.lower()].get("params", {})
            cfg = {**yaml_params, **cfg}

        return model_cls(model_name=name, config=cfg)


# Explicitly register built-in models
from models.sklearn_wrappers.random_forest import RandomForestBlastModel
from models.sklearn_wrappers.xgboost import XGBoostBlastModel
from models.sklearn_wrappers.ridge import RidgeBlastModel
from models.custom.ga_ann import GAANNBlastModel
from models.custom.pinn import PINNBlastModel
from models.custom.ensemble import EnsembleBlastModel
from models.custom.site_calibration_model import SiteCalibrationBlastModel

register_model("random_forest")(RandomForestBlastModel)
register_model("xgboost")(XGBoostBlastModel)
register_model("ridge")(RidgeBlastModel)
register_model("ga_ann")(GAANNBlastModel)
register_model("pinn")(PINNBlastModel)
register_model("ensemble")(EnsembleBlastModel)
register_model("site_calibration")(SiteCalibrationBlastModel)
