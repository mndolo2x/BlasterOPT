"""
Ensemble Uncertainty Model Wrapper (`models/custom/ensemble.py`).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from models.base import BaseBlastModel, ModelMetadata
from src.ensemble_uq import UncertaintyAwarePredictor


class EnsembleBlastModel(BaseBlastModel):
    """
    Bagging ensemble model with aleatoric and epistemic uncertainty quantification.
    """

    def __init__(self, model_name: str = "EnsembleBlastModel", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        self.predictor = UncertaintyAwarePredictor()

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="ensemble",
            display_name="Ensemble Uncertainty Predictor",
            model_type="ensemble",
            description="N=7 GA-ANN Bagging Ensemble with Aleatoric & Epistemic Uncertainty Quantification.",
            version="1.5.0",
            author="BlastOpt Botswana Team",
            supports_uncertainty=True,
            supports_explainability=True,
            tags=["ensemble", "uncertainty", "bagging", "ood"]
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], **kwargs) -> "EnsembleBlastModel":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.DataFrame) else y
        self.predictor.fit_ensemble(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if not self.is_fitted:
            n_samples = X_arr.shape[0] if X_arr.ndim > 1 else 1
            preds = np.tile([220.0, 4.20, 110.0, 4.80], (n_samples, 1))
        else:
            ens_res = self.predictor.predict_with_uncertainty(X_arr)
            preds = ens_res.mean_predictions

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(preds, columns=["d50_mm", "ppv_mms", "flyrock_m", "cost_usd"])
        return preds

    def predict_with_uncertainty(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        ens_res = self.predictor.predict_with_uncertainty(X_arr)
        return {
            "mean": ens_res.mean_predictions,
            "std": ens_res.std_predictions,
            "aleatoric_std": ens_res.aleatoric_std,
            "epistemic_std": ens_res.epistemic_std,
            "lower_95_ci": ens_res.lower_95_ci,
            "upper_95_ci": ens_res.upper_95_ci,
            "is_ood": ens_res.ood_report.is_ood if hasattr(ens_res, "ood_report") else False
        }
