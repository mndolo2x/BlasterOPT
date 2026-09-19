"""
Models Architecture Package (`models`).
Contains BaseBlastModel interface, ModelMetadata, ModelRegistry with auto-discovery,
sklearn wrappers, custom physics & neural network models, and get_registry() singleton.
"""

from models.base import BaseBlastModel, ModelMetadata
from models.registry import ModelRegistry, get_registry, register_model
from models.sklearn_wrappers.random_forest import RandomForestBlastModel
from models.sklearn_wrappers.xgboost import XGBoostBlastModel
from models.sklearn_wrappers.ridge import RidgeBlastModel
from models.custom.ga_ann import GAANNBlastModel
from models.custom.pinn import PINNBlastModel
from models.custom.ensemble import EnsembleBlastModel
from models.custom.site_calibration_model import SiteCalibrationBlastModel

__all__ = [
    "BaseBlastModel",
    "ModelMetadata",
    "ModelRegistry",
    "get_registry",
    "register_model",
    "RandomForestBlastModel",
    "XGBoostBlastModel",
    "RidgeBlastModel",
    "GAANNBlastModel",
    "PINNBlastModel",
    "EnsembleBlastModel",
    "SiteCalibrationBlastModel",
]
