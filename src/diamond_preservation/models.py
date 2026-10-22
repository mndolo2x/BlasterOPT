"""
Data Models and Pydantic Schemas for Diamond Preservation Subsystem.

Defines domain schemas for diamond observations, breakage assessments,
scenarios, valuation schedules, and dataset readiness reports.
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DiamondObservation(BaseModel):
    """
    Record of observed diamond stone or batch-level damage.
    """
    observation_id: str = Field(default_factory=lambda: f"OBS_{int(datetime.now(timezone.utc).timestamp())}", description="Unique observation ID")
    blast_id: str = Field(..., description="Associated blast ID")
    sample_lot_id: str = Field(..., description="Sample or lot identifier")
    stage: Literal[
        "before_blasting", "post_blast", "excavation_loading",
        "transport", "crushing", "milling", "recovery", "unknown"
    ] = Field("post_blast", description="Process stage where observation occurred")
    observation_method: str = Field("Visual / Optical Scanner", description="Method used")
    stone_size_class: str = Field("+11 DTC", description="Size category (e.g. +11 DTC, +9 DTC, Special > 10.8ct)")
    stone_count: int = Field(1, description="Number of stones in observation")
    total_carats: float = Field(..., description="Total weight in carats")
    stone_quality_category: str = Field("Gem", description="Quality or value category")
    damage_type: Literal[
        "none", "chipping", "cracking", "breakage_cleavage", "severe_fragmentation", "unknown"
    ] = Field("none", description="Observed damage type")
    damage_attribution_cause: Literal[
        "blast_induced", "crusher_induced", "natural_flaw", "unknown"
    ] = Field("unknown", description="Attributed cause of damage")
    attribution_confidence: float = Field(0.8, description="Attribution confidence score (0.0 to 1.0)")
    observer_id: str = Field("GEOLOGIST_BW_01", description="Observer or team ID")
    observation_date: str = Field(default_factory=_utc_now_iso, description="Date of observation")
    notes: Optional[str] = Field(None, description="Notes or limitations")


class DiamondAssessment(BaseModel):
    """
    Assessment record linking blast inputs, geomechanics, and breakage evaluation.
    """
    assessment_id: str = Field(..., description="Unique assessment ID")
    blast_id: str = Field(..., description="Associated blast ID")
    assessment_timestamp: str = Field(default_factory=_utc_now_iso, description="ISO timestamp")
    assessor_id: str = Field("MINING_ENGINEER_01", description="Engineer ID")
    completeness_score_pct: float = Field(..., description="Data completeness percentage")
    missing_mandatory_fields: List[str] = Field(default_factory=list, description="Missing required inputs")
    missing_optional_fields: List[str] = Field(default_factory=list, description="Missing optional inputs")
    specific_charge_kg_t: Optional[float] = Field(None, description="Calculated specific charge (kg/t)")
    max_charge_per_delay_kg: Optional[float] = Field(None, description="Max charge per delay (kg)")
    vibration_ppv_mms: Optional[float] = Field(None, description="Measured or predicted PPV")
    model_prediction_status: str = Field(
        "Diamond-breakage prediction unavailable — no validated model is configured.",
        description="Status message of predictor"
    )
    risk_category: Literal["Not assessed", "Low Risk", "Moderate Risk", "High Risk"] = Field(
        "Not assessed", description="Evaluated risk category"
    )
    input_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Snapshot of inputs used")
    provenance: str = Field("BlastOPT Diamond Module", description="Assessment provenance")


class DiamondScenario(BaseModel):
    """
    What-if design scenario comparing candidate parameters against a baseline blast.
    """
    scenario_id: str = Field(..., description="Unique scenario ID")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    baseline_blast_id: str = Field(..., description="Source baseline blast ID")
    created_at: str = Field(default_factory=_utc_now_iso, description="Creation timestamp")
    created_by: str = Field("ENGINEER_01", description="Author ID")
    candidate_params: Dict[str, Any] = Field(default_factory=dict, description="Candidate blast parameters")
    parameter_deltas: Dict[str, Any] = Field(default_factory=dict, description="Parameter differences vs baseline")
    feasibility_status: Literal["Verified", "Unverified", "Rejected"] = Field("Unverified", description="Feasibility state")
    warning_messages: List[str] = Field(default_factory=list, description="Warnings or rejection reasons")


class DiamondValuationSchedule(BaseModel):
    """
    Valuation schedule by stone size and quality class ($/carat).
    """
    schedule_id: str = Field(..., description="Unique valuation schedule ID")
    schedule_name: str = Field(..., description="Schedule name (e.g. Q4 2026 Debswana Diamond Price Index)")
    effective_date: str = Field(default_factory=_utc_now_iso, description="Effective ISO date")
    currency: str = Field("USD", description="Currency code")
    valuation_map: Dict[str, float] = Field(
        default_factory=lambda: {
            "Special (>10.8ct)": 2500.0,
            "+11 DTC Gem": 450.0,
            "+11 DTC Near Gem": 180.0,
            "+9 DTC Gem": 280.0,
            "+9 DTC Near Gem": 110.0,
            "-9 DTC Rejection": 25.0,
        },
        description="Price per carat by size/quality class"
    )
    source: str = Field("Debswana Diamond Price Index", description="Valuation source")


class DiamondDatasetReadinessReport(BaseModel):
    """
    Dataset readiness report for future predictive model training.
    """
    total_observations: int = Field(..., description="Total diamond observation records")
    total_distinct_blasts: int = Field(..., description="Distinct blasts represented")
    usable_independent_blasts: int = Field(..., description="Independent blasts with complete labels")
    label_completeness_pct: float = Field(..., description="Percentage of records with non-unknown damage labels")
    domain_coverage: Dict[str, int] = Field(default_factory=dict, description="Counts by geological domain")
    stage_breakdown: Dict[str, int] = Field(default_factory=dict, description="Counts by observation stage")
    is_ready_for_training: bool = Field(False, description="Whether dataset meets minimum readiness threshold")
    readiness_warnings: List[str] = Field(default_factory=list, description="Readiness warnings or data gaps")
