"""
Pydantic Data Models for Ensemble Uncertainty Quantification & OOD Detection Module.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration settings for GA-ANN bagging ensemble training."""
    n_models: int = Field(7, ge=3, le=50, description="Number of GA-ANN models in ensemble")
    architectures: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"hidden_layer_sizes": (64, 32), "activation": "relu"},
            {"hidden_layer_sizes": (128, 64), "activation": "tanh"},
            {"hidden_layer_sizes": (32, 32, 16), "activation": "relu"},
        ],
        description="List of neural network architecture specifications"
    )
    seeds: List[int] = Field(
        default_factory=lambda: [42, 101, 202, 303, 404, 505, 606],
        description="Random seeds for ensemble member initialization"
    )
    data_splits: float = Field(0.80, ge=0.50, le=0.95, description="Fraction of bootstrap training sample per member")


class OODReport(BaseModel):
    """Report detailing Out-Of-Distribution (OOD) feature detection metrics."""
    input_features: Dict[str, float] = Field(..., description="Feature key-value map of input sample")
    distances_to_training: Dict[str, float] = Field(..., description="Mahalanobis and Euclidean distance to training distribution")
    threshold_exceeded: List[str] = Field(default_factory=list, description="List of specific range or distance thresholds exceeded")
    is_ood: bool = Field(False, description="Whether the sample is classified as Out-Of-Distribution")
    severity: str = Field("NORMAL", description="OOD severity level: 'NORMAL', 'MODERATE', 'HIGH', 'CRITICAL'")
    reason: str = Field("In-distribution training parameters", description="Detailed natural language explanation of OOD status")


class EnsemblePrediction(BaseModel):
    """Structured prediction result with decomposed uncertainty and OOD assessment."""
    mean: Dict[str, float] = Field(..., description="Ensemble mean predicted values across targets (d50_mm, ppv_mms, flyrock_m, cost)")
    std: Dict[str, float] = Field(..., description="Ensemble standard deviation across target predictions")
    lower_95: Dict[str, float] = Field(..., description="Lower 95% confidence bound (mean - 1.96 * std)")
    upper_95: Dict[str, float] = Field(..., description="Upper 95% confidence bound (mean + 1.96 * std)")
    aleatoric: Dict[str, float] = Field(..., description="Aleatoric (data noise) uncertainty component variance")
    epistemic: Dict[str, float] = Field(..., description="Epistemic (model knowledge gap) uncertainty component variance")
    is_ood: bool = Field(False, description="Whether input is flagged as Out-Of-Distribution")
    ood_reason: str = Field("In-distribution sample", description="Reason for OOD classification or safety warning")
    ood_report: Optional[OODReport] = Field(None, description="Detailed OOD report object")
