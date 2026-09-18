"""
Uncertainty Quantifier Submodule.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.ensemble_uq.models import EnsemblePrediction, OODReport
from src.ensemble_uq.ood_detector import OODDetector

logger = logging.getLogger(__name__)

TARGET_NAMES = ["d50_mm", "ppv_mms", "flyrock_m", "cost_per_tonne_usd"]


class UncertaintyQuantifier:
    """
    Computes ensemble predictions, standard deviation, 95% confidence intervals,
    and decomposes total uncertainty into aleatoric (data noise) and epistemic (model variance).
    """

    def __init__(self, ood_detector: Optional[OODDetector] = None):
        self.ood_detector = ood_detector or OODDetector()

    def compute_uncertainty(
        self,
        ensemble_predictions: np.ndarray,
        feature_dict: Dict[str, float]
    ) -> EnsemblePrediction:
        """
        Calculates mean, std, 95% CI, and aleatoric/epistemic decomposition.

        Parameters:
        -----------
        ensemble_predictions : np.ndarray
            Shape (N_members, N_targets) prediction matrix from N_members.
        feature_dict : Dict[str, float]
            Input feature dictionary for OOD evaluation.
        """
        # Mean across ensemble members
        means_arr = np.mean(ensemble_predictions, axis=0)
        # Epistemic uncertainty = variance across ensemble members
        epistemic_var = np.var(ensemble_predictions, axis=0)
        stds_arr = np.sqrt(epistemic_var)

        # Aleatoric uncertainty estimate (estimated base data noise ~ 5% of mean)
        aleatoric_var = (means_arr * 0.05) ** 2

        # 95% Confidence Interval: Mean ± 1.96 * std
        lower_95_arr = np.maximum(0.01, means_arr - 1.96 * stds_arr)
        upper_95_arr = means_arr + 1.96 * stds_arr

        # Format dictionaries per target metric
        target_keys = TARGET_NAMES[:len(means_arr)] if len(means_arr) <= len(TARGET_NAMES) else [f"target_{i}" for i in range(len(means_arr))]

        means_dict = {k: round(float(v), 2) for k, v in zip(target_keys, means_arr)}
        stds_dict = {k: round(float(v), 2) for k, v in zip(target_keys, stds_arr)}
        lower_dict = {k: round(float(v), 2) for k, v in zip(target_keys, lower_95_arr)}
        upper_dict = {k: round(float(v), 2) for k, v in zip(target_keys, upper_95_arr)}
        aleatoric_dict = {k: round(float(v), 3) for k, v in zip(target_keys, aleatoric_var)}
        epistemic_dict = {k: round(float(v), 3) for k, v in zip(target_keys, epistemic_var)}

        # Evaluate OOD status
        ood_rep = self.ood_detector.detect(feature_dict)

        return EnsemblePrediction(
            mean=means_dict,
            std=stds_dict,
            lower_95=lower_dict,
            upper_95=upper_dict,
            aleatoric=aleatoric_dict,
            epistemic=epistemic_dict,
            is_ood=ood_rep.is_ood,
            ood_reason=ood_rep.reason,
            ood_report=ood_rep,
        )
