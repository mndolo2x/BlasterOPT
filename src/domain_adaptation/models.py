"""
Pydantic Data Models for Geology Domain Adaptation Package (`domain_adaptation`).
"""

import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DomainAdaptationConfig(BaseModel):
    """Configuration options for transfer learning and domain adaptation."""
    source_geology: str = Field(default="Kimberlite", description="Source domain geology type")
    target_geology: str = Field(default="Granite", description="Target domain geology type")
    freeze_early_layers: bool = Field(default=True, description="Whether to freeze early feature extractor layers during fine-tuning")
    fine_tune_lr: float = Field(default=0.0001, description="Learning rate for transfer learning fine-tuning")
    fine_tune_epochs: int = Field(default=50, description="Epochs for transfer learning fine-tuning")
    enable_jda: bool = Field(default=True, description="Whether to enable Joint Domain Adaptation (JDA)")
    enable_dann: bool = Field(default=False, description="Whether to enable Domain-Adversarial Neural Network (DANN) adaptation")
    target_r2_threshold: float = Field(default=0.75, description="Target domain R2 threshold below which adversarial adaptation is triggered")
    alpha_grl: float = Field(default=1.0, description="Gradient Reversal Layer (GRL) scaling factor for DANN")


class DomainShiftMetrics(BaseModel):
    """Metrics measuring distribution shift between source and target domains."""
    source_domain: str = Field(..., description="Source geology name")
    target_domain: str = Field(..., description="Target geology name")
    mmd_distance: float = Field(..., description="Maximum Mean Discrepancy (MMD) feature distance")
    domain_shift_level: str = Field(..., description="Shift level: 'LOW', 'MODERATE', or 'HIGH'")


class FineTuneMetrics(BaseModel):
    """Evaluation metrics for target domain model fine-tuning."""
    target_geology: str = Field(..., description="Target geology name")
    sample_count: int = Field(..., description="Target domain training sample count")
    pre_adaptation_r2: float = Field(..., description="Model R2 on target domain before fine-tuning")
    post_adaptation_r2: float = Field(..., description="Model R2 on target domain after fine-tuning")
    r2_improvement: float = Field(..., description="R2 score improvement")
    target_rmse: float = Field(..., description="RMSE on target domain after fine-tuning")
    target_mae: float = Field(..., description="MAE on target domain after fine-tuning")
    method_used: str = Field(..., description="Adaptation method used: 'FINE_TUNING', 'JDA_ALIGNMENT', or 'DANN_ADVERSARIAL'")


class DomainAdaptationReport(BaseModel):
    """Overall summary report for domain adaptation execution."""
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(), description="ISO UTC timestamp")
    source_domain: str = Field(..., description="Source geology domain")
    target_domain: str = Field(..., description="Target geology domain")
    status: str = Field(..., description="Adaptation status: 'ADAPTATION_SUCCESSFUL', 'THRESHOLD_EXCEEDED', or 'INSUFFICIENT_SAMPLES'")
    domain_shift: DomainShiftMetrics = Field(..., description="Distribution shift metrics")
    metrics: FineTuneMetrics = Field(..., description="Adaptation metrics summary")
    recommendations: List[str] = Field(default_factory=list, description="Engineering recommendations for blast design in target geology")
