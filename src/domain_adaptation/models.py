"""
Pydantic Data Models for Geology Domain Adaptation Package (`domain_adaptation`).
"""

import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class DomainData(BaseModel):
    """Container for source and target domain features and target labels."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_domain: str = Field(default="Kimberlite", description="Source geology name")
    target_domain: str = Field(default="Granite", description="Target geology name")
    features: Union[np.ndarray, List[List[float]]] = Field(..., description="Feature matrix X")
    labels: Optional[Union[np.ndarray, List[List[float]]]] = Field(default=None, description="Target array Y")


class AdaptationResult(BaseModel):
    """Result of domain adaptation execution."""
    method: str = Field(..., description="Adaptation method: FINE_TUNE, JDA, or DANN")
    source_r2: float = Field(..., description="Model R2 score on source domain")
    target_r2: float = Field(..., description="Model R2 score on target domain after adaptation")
    improvement: float = Field(..., description="Target domain R2 score improvement over zero-shot baseline")
    target_rmse: Optional[float] = Field(default=None, description="Target domain RMSE")
    target_mae: Optional[float] = Field(default=None, description="Target domain MAE")


class DomainShiftReport(BaseModel):
    """Report measuring distribution shift between reference and new domain data."""
    source_domain: str = Field(default="Kimberlite", description="Reference/source geology")
    target_domain: str = Field(default="Granite", description="New/target geology")
    mmd_score: float = Field(..., description="Maximum Mean Discrepancy (MMD) distance score")
    wasserstein_distance: float = Field(..., description="Mean 1D Wasserstein distance across feature dimensions")
    feature_drift: Dict[str, float] = Field(default_factory=dict, description="Feature-level distribution shift scores")
    shift_level: str = Field(..., description="Overall shift level: 'LOW', 'MODERATE', or 'HIGH'")


# Alias for backward compatibility
DomainShiftMetrics = DomainShiftReport


class FineTuneMetrics(BaseModel):
    """Evaluation metrics for target domain model fine-tuning."""
    target_geology: str = Field(..., description="Target geology name")
    sample_count: int = Field(..., description="Target domain training sample count")
    pre_adaptation_r2: float = Field(..., description="Model R2 on target domain before fine-tuning")
    post_adaptation_r2: float = Field(..., description="Model R2 on target domain after fine-tuning")
    r2_improvement: float = Field(..., description="R2 score improvement")
    target_rmse: float = Field(..., description="RMSE on target domain after fine-tuning")
    target_mae: float = Field(..., description="MAE on target domain after fine-tuning")
    method_used: str = Field(..., description="Adaptation method used: 'FINE_TUNE', 'JDA', or 'DANN'")


class DomainAdaptationReport(BaseModel):
    """Overall summary report for domain adaptation execution."""
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(), description="ISO UTC timestamp")
    source_domain: str = Field(..., description="Source geology domain")
    target_domain: str = Field(..., description="Target geology domain")
    status: str = Field(..., description="Adaptation status: 'ADAPTATION_SUCCESSFUL', 'THRESHOLD_EXCEEDED', or 'INSUFFICIENT_SAMPLES'")
    domain_shift: DomainShiftReport = Field(..., description="Distribution shift metrics")
    metrics: FineTuneMetrics = Field(..., description="Adaptation metrics summary")
    recommendations: List[str] = Field(default_factory=list, description="Engineering recommendations for blast design in target geology")


class DomainAdaptationConfig(BaseModel):
    """Configuration options for domain adaptation algorithms."""
    source_geology: str = Field(default="Kimberlite", description="Source domain geology type")
    target_geology: str = Field(default="Granite", description="Target domain geology type")
    freeze_depth: int = Field(default=2, description="Number of early network layers to freeze during fine-tuning")
    fine_tune_lr: float = Field(default=0.001, description="Learning rate for fine-tuning")
    fine_tune_epochs: int = Field(default=60, description="Epochs for fine-tuning")
    alpha_grl: float = Field(default=1.0, description="Gradient Reversal Layer (GRL) scaling factor for DANN")
