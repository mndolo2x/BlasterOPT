"""
Physics-Informed Neural Network Package for BlasterOPT Botswana.
"""

from src.physics_informed.physics_equations import (
    kuz_ram_x50_torch,
    usbm_ppv_torch,
    kuz_ram_x50_numpy,
    usbm_ppv_numpy,
)
from src.physics_informed.pinn_model import PhysicsInformedGAANN
from src.physics_informed.loss import CompositePhysicsLoss
from src.physics_informed.trainer import PINNTrainer
from src.physics_informed.evaluator import PINNEvaluator
from src.physics_informed.visualizer import plot_physics_loss_curves, plot_extrapolation_comparison

__all__ = [
    "kuz_ram_x50_torch",
    "usbm_ppv_torch",
    "kuz_ram_x50_numpy",
    "usbm_ppv_numpy",
    "PhysicsInformedGAANN",
    "CompositePhysicsLoss",
    "PINNTrainer",
    "PINNEvaluator",
    "plot_physics_loss_curves",
    "plot_extrapolation_comparison",
]
