"""
Pydantic Data Models for Site-Specific Attenuation Calibration Module.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class SeismographReading(BaseModel):
    """Represents a single seismograph vibration measurement from a blast event."""
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp of blast event")
    blast_id: str = Field(..., description="Unique blast event identifier")
    site_id: str = Field(..., description="Mining site or pit zone identifier")
    distance_m: float = Field(..., gt=0, description="Horizontal distance from blast site to seismograph in meters")
    ppv_mm_s: float = Field(..., gt=0, description="Measured Peak Particle Velocity ground vibration in mm/s")
    charge_per_delay_kg: float = Field(..., gt=0, description="Maximum explosive charge mass detonated per delay interval (< 8ms) in kg")
    dominant_frequency_hz: Optional[float] = Field(None, gt=0, description="Dominant vibration wave frequency in Hz")
    gsi_of_transmission_strata: Optional[float] = Field(None, ge=0, le=100, description="Geological Strength Index (GSI) of rock mass transmission path")
    rock_type: Optional[str] = Field("Kimberlite", description="Dominant rock type of transmission strata")


class AttenuationParameters(BaseModel):
    """Fitted USBM and GSI-modified attenuation law parameters."""
    K: float = Field(..., description="Site attenuation site constant K")
    B: float = Field(..., description="Site attenuation exponent B")
    gsi_a: Optional[float] = Field(1.0, description="GSI exponential correction scaling factor 'a'")
    gsi_b: Optional[float] = Field(0.0, description="GSI exponential correction rate factor 'b'")
    r_squared: float = Field(..., ge=0.0, le=1.0, description="Coefficient of determination (R2) score")
    rmse: float = Field(..., ge=0.0, description="Root Mean Squared Error (RMSE) in mm/s")
    confidence_interval_95: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="95% confidence intervals for fitted parameters")
    sample_count: int = Field(..., ge=0, description="Number of seismograph readings used in fit")
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp of calibration fit")
    rock_type: str = Field("Kimberlite", description="Rock type associated with parameter fit")


class CalibrationResult(BaseModel):
    """Result summary of a site calibration fitting run."""
    site_id: str = Field(..., description="Mining site or zone identifier")
    rock_type: str = Field("Kimberlite", description="Rock type associated with calibration")
    parameters: Optional[AttenuationParameters] = Field(None, description="Fitted attenuation parameters")
    fit_quality: str = Field(..., description="Fit quality rating: 'EXCELLENT', 'ACCEPTABLE', 'POOR', 'REJECTED'")
    warnings: List[str] = Field(default_factory=list, description="List of warnings or rejection reasons")
    recommendation: str = Field(..., description="Actionable recommendation for blast engineering team")


class PPVPrediction(BaseModel):
    """Predicted Peak Particle Velocity (PPV) with 95% confidence bounds."""
    ppv_mm_s: float = Field(..., description="Predicted PPV ground vibration in mm/s")
    lower_95: float = Field(..., description="Lower 95% confidence bound in mm/s")
    upper_95: float = Field(..., description="Upper 95% confidence bound in mm/s")
    method: str = Field(..., description="Prediction methodology: 'USBM_SITE_CALIBRATED', 'GSI_MODIFIED_USBM', 'USBM_GENERIC_FALLBACK'")
    parameters_used: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of parameter values used")


class IngestionReport(BaseModel):
    """Report summarizing seismograph data ingestion results."""
    accepted_count: int = Field(0, description="Number of successfully ingested valid readings")
    rejected_count: int = Field(0, description="Number of rejected invalid readings")
    rejected_rows: List[Dict[str, Any]] = Field(default_factory=list, description="List of rejected row dicts with rejection reason")


class FitQualityReport(BaseModel):
    """Summary of site calibration quality metrics."""
    site_id: str = Field(..., description="Mining site identifier")
    rock_type: str = Field("Kimberlite", description="Rock type")
    r_squared: float = Field(..., description="R2 fit score")
    rmse: float = Field(..., description="RMSE fit error")
    sample_count: int = Field(..., description="Number of data points")
    is_statistically_sound: bool = Field(..., description="Whether fit meets R2 >= 0.70 and sample >= 10 bounds")
    status: str = Field(..., description="Quality status description")


class RecalibrationLog(BaseModel):
    """Log entry for continuous feedback loop recalibration events."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp")
    site_id: str = Field(..., description="Site identifier")
    rock_type: str = Field(..., description="Rock type")
    sample_count: int = Field(..., description="Sample count used in recalibration")
    old_params: Dict[str, float] = Field(..., description="Parameters prior to recalibration")
    new_params: Dict[str, float] = Field(..., description="Parameters after recalibration")
    delta_k_pct: float = Field(..., description="Percentage change in K parameter")
    delta_b_pct: float = Field(..., description="Percentage change in B parameter")
    reason: str = Field(..., description="Trigger reason for recalibration")
    alerts: List[str] = Field(default_factory=list, description="Alert messages generated during recalibration")
