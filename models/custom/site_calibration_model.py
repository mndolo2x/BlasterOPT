"""
Site-Specific Attenuation Calibration Model Wrapper (`models/custom/site_calibration_model.py`).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from models.base import BaseBlastModel, ModelMetadata
from src.site_calibration import CalibrationManager


class SiteCalibrationBlastModel(BaseBlastModel):
    """
    Site-Specific Vibration Attenuation Model with GSI Transmission Correction.
    """

    def __init__(self, model_name: str = "SiteCalibrationBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        self.site_id = self.config.get("site_id", "Jwaneng_Main_Pit")
        self.manager = CalibrationManager()

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="site_calibration",
            display_name="Site-Specific Vibration Calibration Model",
            model_type="calibration",
            description="USBM & GSI-modified Vibration Wave Attenuation Calibration Engine.",
            version="1.2.0",
            author="BlastOpt Botswana Team",
            supports_uncertainty=True,
            tags=["calibration", "usbm", "vibration", "ppv"]
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "SiteCalibrationBlastModel":
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = X_arr.shape[0] if X_arr.ndim > 1 else 1
        preds = np.zeros((n_samples, 4))
        preds[:, 0] = 220.0  # d50
        preds[:, 2] = 110.0  # flyrock
        preds[:, 3] = 4.80   # cost

        for i, row in enumerate(X_arr):
            d_val = row[10] if len(row) > 10 else 450.0
            q_val = row[7] if len(row) > 7 else 320.0
            ppv_pred = self.manager.predict_ppv(self.site_id, d_val, q_val)
            preds[i, 1] = ppv_pred.predicted_ppv_mms

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(preds, columns=["d50_mm", "ppv_mms", "flyrock_m", "cost_usd"])
        return preds
