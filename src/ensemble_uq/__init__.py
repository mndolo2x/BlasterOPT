"""
Ensemble Uncertainty Quantification & OOD Detection Package for BlasterOPT Botswana.
"""

from src.ensemble_uq.models import TrainingConfig, OODReport, EnsemblePrediction
from src.ensemble_uq.ensemble_trainer import GAANNEnsembleMember, EnsembleTrainer
from src.ensemble_uq.ood_detector import OODDetector
from src.ensemble_uq.uncertainty_quantifier import UncertaintyQuantifier
from src.ensemble_uq.prediction import UncertaintyAwarePredictor
from src.ensemble_uq.visualizer import (
    plot_prediction_intervals,
    plot_uncertainty_decomposition,
    plot_ood_distribution,
    plot_confidence_calibration,
)

__all__ = [
    "TrainingConfig",
    "OODReport",
    "EnsemblePrediction",
    "GAANNEnsembleMember",
    "EnsembleTrainer",
    "OODDetector",
    "UncertaintyQuantifier",
    "UncertaintyAwarePredictor",
    "plot_prediction_intervals",
    "plot_uncertainty_decomposition",
    "plot_ood_distribution",
    "plot_confidence_calibration",
]
