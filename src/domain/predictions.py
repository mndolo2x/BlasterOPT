"""
Pure Model Predictions Domain Module for BlastOpt Botswana.

Defines Pydantic models for raw model predictions with uncertainty bounds (mean, lower_95, upper_95, std).
Strictly contains model predictions only; contains NO recommendation or decision logic.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class PredictionValue(BaseModel):
    """
    Individual predicted output variable with 95% uncertainty bounds.
    """
    mean: float = Field(..., description="Predicted mean value")
    lower_95: float = Field(..., description="Lower 95% confidence bound")
    upper_95: float = Field(..., description="Upper 95% confidence bound")
    std: Optional[float] = Field(None, description="Standard deviation / epistemic uncertainty")
    unit: str = Field(..., description="Physical unit of measurement (e.g., mm/s, dBL, m, $/t)")


class BlastPrediction(BaseModel):
    """
    Pure model prediction outputs with uncertainty intervals for a single blast design input.
    Contains NO recommendation or approval logic.
    """
    d50_mm: PredictionValue = Field(..., description="Mean fragmentation size d50")
    ppv_mms: PredictionValue = Field(..., description="Ground vibration peak particle velocity")
    airblast_dbl: PredictionValue = Field(..., description="Airblast noise overpressure")
    flyrock_m: PredictionValue = Field(..., description="Maximum flyrock throw distance")
    cost_per_tonne_usd: PredictionValue = Field(..., description="Total D&B unit cost")
    raw_outputs: Optional[Dict[str, float]] = Field(None, description="Dictionary of mean prediction scalar values")
