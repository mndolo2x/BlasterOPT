"""
Models Architecture Package (`models`).
Contains BaseBlastModel interface, ModelRegistry with auto-discovery,
sklearn wrappers, custom physics & neural network models, and YAML configs.
"""

from models.base import BaseBlastModel
from models.registry import ModelRegistry, register_model
from models.sklearn_wrappers.random_forest import RandomForestBlastModel
from models.sklearn_wrappers.xgboost import XGBoostBlastModel
from models.sklearn_wrappers.ridge import RidgeBlastModel
from models.custom.ga_ann import GAANNBlastModel
from models.custom.pinn import PINNBlastModel
from models.custom.ensemble import EnsembleBlastModel
from models.custom.site_calibration_model import SiteCalibrationBlastModel

__all__ = [
    "BaseBlastModel",
    "ModelRegistry",
    "register_model",
    "RandomForestBlastModel",
    "XGBoostBlastModel",
    "RidgeBlastModel",
    "GAANNBlastModel",
    "PINNBlastModel",
    "EnsembleBlastModel",
    "SiteCalibrationBlastModel",
]
