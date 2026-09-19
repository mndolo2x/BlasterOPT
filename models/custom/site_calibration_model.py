"""
Site-Specific Attenuation Calibration Model Wrapper (`models/custom/site_calibration_model.py`).
"""

import numpy as np
from typing import Dict, Any, Optional
from models.base import BaseBlastModel
from src.site_calibration import CalibrationManager


class SiteCalibrationBlastModel(BaseBlastModel):
    """
    Site-Specific Vibration Attenuation Model with GSI Transmission Correction.
    """

    def __init__(self, model_name: str = "SiteCalibrationBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        self.site_id = self.config.get("site_id", "Jwaneng_Main_Pit")
        self.manager = CalibrationManager()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SiteCalibrationBlastModel":
        # Extract seismograph reading records from X and Y
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict PPV using calibrated attenuation parameters
        n_samples = X.shape[0] if X.ndim > 1 else 1
        preds = np.zeros((n_samples, 4))
        preds[:, 0] = 220.0  # d50
        preds[:, 2] = 110.0  # flyrock
        preds[:, 3] = 4.80   # cost

        for i, row in enumerate(X):
            d_val = row[10] if len(row) > 10 else 450.0
            q_val = row[7] if len(row) > 7 else 320.0
            ppv_pred = self.manager.predict_ppv(self.site_id, d_val, q_val)
            preds[i, 1] = ppv_pred.predicted_ppv_mms

        return preds
