"""
Out-Of-Distribution (OOD) Detector Submodule.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from src.ensemble_uq.models import OODReport

logger = logging.getLogger(__name__)

# Configurable domain feature validity bounds
DEFAULT_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "powder_factor_kg_m3": (0.20, 1.20),
    "bench_height_m": (5.0, 18.0),
    "burden_m": (2.0, 10.0),
    "spacing_m": (2.0, 12.0),
    "stemming_m": (1.0, 8.0),
    "hole_diameter_mm": (80.0, 380.0),
    "charge_mass_per_hole_kg": (10.0, 1200.0),
    "max_charge_per_delay_kg": (10.0, 2500.0),
    "monitoring_distance_m": (50.0, 2500.0),
}


class OODDetector:
    """
    Detects Out-Of-Distribution (OOD) inputs using three complementary techniques:
    1. Range checks against explicit domain physics bounds (e.g. powder factor > 1.20 kg/m³, bench height > 18m).
    2. Mahalanobis distance relative to training distribution covariance.
    3. Isolation Forest density anomaly detection.
    """

    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.bounds = bounds or DEFAULT_FEATURE_BOUNDS
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.mean_vec: Optional[np.ndarray] = None
        self.inv_cov_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "OODDetector":
        """
        Fits Isolation Forest and Mahalanobis covariance matrix on in-distribution training data.
        """
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        self.iso_forest.fit(X)

        self.mean_vec = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
        self.inv_cov_matrix = np.linalg.pinv(cov)
        self.is_fitted = True
        return self

    def calculate_mahalanobis_distance(self, x_vec: np.ndarray) -> float:
        """Calculates Mahalanobis distance for a single feature vector."""
        if not self.is_fitted or self.mean_vec is None or self.inv_cov_matrix is None:
            return 0.0
        diff = x_vec - self.mean_vec
        return float(np.sqrt(np.dot(np.dot(diff, self.inv_cov_matrix), diff.T)))

    def detect(self, feature_dict: Dict[str, float]) -> OODReport:
        """
        Evaluates a feature dictionary for OOD violations and returns an OODReport.
        """
        exceeded_thresholds: List[str] = []

        # 1. Range bounds verification
        for feat_name, (min_val, max_val) in self.bounds.items():
            if feat_name in feature_dict:
                val = float(feature_dict[feat_name])
                if val < min_val:
                    exceeded_thresholds.append(f"'{feat_name}' value {val:.2f} is below minimum bound ({min_val:.2f})")
                elif val > max_val:
                    exceeded_thresholds.append(f"'{feat_name}' value {val:.2f} exceeds maximum bound ({max_val:.2f})")

        # 2. Mahalanobis distance & Isolation Forest checks if feature vector matches
        distances = {"mahalanobis_distance": 0.0, "isolation_forest_score": 0.0}
        iso_anomaly = False

        if self.is_fitted and self.feature_names:
            x_arr = np.array([feature_dict.get(fn, 0.0) for fn in self.feature_names])
            m_dist = self.calculate_mahalanobis_distance(x_arr)
            iso_score = float(self.iso_forest.score_samples(x_arr.reshape(1, -1))[0])
            distances["mahalanobis_distance"] = round(m_dist, 2)
            distances["isolation_forest_score"] = round(iso_score, 3)

            # High Mahalanobis distance > 4.5 or Isolation Forest score < -0.15
            if m_dist > 4.5:
                exceeded_thresholds.append(f"Mahalanobis distance ({m_dist:.2f}) exceeds statistical bound (4.50)")
            if iso_score < -0.15:
                iso_anomaly = True
                exceeded_thresholds.append(f"Isolation Forest score ({iso_score:.3f}) indicates low-density OOD region")

        is_ood = len(exceeded_thresholds) > 0

        # Severity classification
        if not is_ood:
            severity = "NORMAL"
            reason = "Feature vector is within in-distribution training bounds."
        elif len(exceeded_thresholds) == 1 and "Mahalanobis" not in exceeded_thresholds[0]:
            severity = "MODERATE"
            reason = f"Out-of-distribution parameter detected: {exceeded_thresholds[0]}."
        elif len(exceeded_thresholds) <= 2:
            severity = "HIGH"
            reason = f"Out-of-distribution parameters flagged: {' | '.join(exceeded_thresholds)}."
        else:
            severity = "CRITICAL"
            reason = f"Critical OOD extrapolation! Multiple features exceed training bounds: {' | '.join(exceeded_thresholds)}."

        return OODReport(
            input_features={k: float(v) for k, v in feature_dict.items()},
            distances_to_training=distances,
            threshold_exceeded=exceeded_thresholds,
            is_ood=is_ood,
            severity=severity,
            reason=reason,
        )
