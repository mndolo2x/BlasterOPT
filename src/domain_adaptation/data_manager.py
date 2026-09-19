"""
Data Management Submodule for Domain Adaptation.
Handles source (kimberlite) and target (granite) feature loading, alignment, and domain labels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class DomainDataManager:
    """
    Manages source (Kimberlite) and target (Granite) blast datasets for domain adaptation.
    """

    def __init__(self, feature_names: Optional[list] = None):
        self.feature_names = feature_names or [
            "burden_m", "spacing_m", "hole_diameter_mm", "bench_height_m",
            "stemming_m", "subdrill_m", "powder_factor_kg_m3", "max_charge_per_delay_kg",
            "rock_factor_a", "rmr", "monitoring_distance_m", "explosive_rws"
        ]

    def create_synthetic_domain_datasets(
        self,
        n_source: int = 200,
        n_target: int = 40,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates synthetic source (Kimberlite) and target (Granite) feature and target arrays.
        Granite rock factor A is harder (~11.0 vs 8.0) and lower RMR (~50 vs 65).
        """
        np.random.seed(seed)

        # Source domain: Kimberlite (Softer rock, A=8.0, RMR=65)
        X_source = np.random.uniform(1.0, 10.0, size=(n_source, len(self.feature_names)))
        X_source[:, 2] = 250.0                                          # Hole diameter
        X_source[:, 6] = np.random.uniform(0.35, 0.85, size=n_source)  # Powder factor
        X_source[:, 7] = np.random.uniform(200.0, 700.0, size=n_source)# Charge
        X_source[:, 8] = 8.0                                            # Rock Factor A (Kimberlite)
        X_source[:, 9] = 65.0                                           # RMR (Kimberlite)
        X_source[:, 10] = np.random.uniform(100.0, 800.0, size=n_source)# Distance

        # Kimberlite targets: [d50_mm, ppv_mms, flyrock_m, cost_usd]
        d50_src = 240.0 - 120.0 * X_source[:, 6] + np.random.normal(0, 2, n_source)
        ppv_src = 100.0 - 0.1 * X_source[:, 10] + np.random.normal(0, 1, n_source)
        Y_source = np.column_stack([d50_src, ppv_src, 80.0 + 30.0 * X_source[:, 6], 3.5 + 1.8 * X_source[:, 6]])

        # Target domain: Granite (Harder rock, A=11.0, RMR=50, higher vibration velocity & larger fragments)
        X_target = np.random.uniform(1.0, 10.0, size=(n_target, len(self.feature_names)))
        X_target[:, 2] = 250.0
        X_target[:, 6] = np.random.uniform(0.50, 1.10, size=n_target)  # Higher powder factor required
        X_target[:, 7] = np.random.uniform(300.0, 900.0, size=n_target)
        X_target[:, 8] = 11.0                                           # Rock Factor A (Granite - harder)
        X_target[:, 9] = 50.0                                           # RMR (Granite)
        X_target[:, 10] = np.random.uniform(100.0, 800.0, size=n_target)

        # Granite targets: Harder rock produces larger fragments for same PF
        d50_tgt = 320.0 - 140.0 * X_target[:, 6] + np.random.normal(0, 2, n_target)
        ppv_tgt = 120.0 - 0.12 * X_target[:, 10] + np.random.normal(0, 1, n_target)
        Y_target = np.column_stack([d50_tgt, ppv_tgt, 90.0 + 35.0 * X_target[:, 6], 4.2 + 2.2 * X_target[:, 6]])

        return X_source, Y_source, X_target, Y_target
