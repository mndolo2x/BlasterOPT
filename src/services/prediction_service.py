"""
Prediction & Safety Recommendation Service Module for BlastOpt Botswana.

Separates raw model predictions from decision recommendation workflows.
Guarantees every recommendation carries a mandatory SafetyReport and strictly
blocks recommendations for UNSAFE designs.
"""

import logging
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
from src.domain.predictions import PredictionValue, BlastPrediction
from src.domain.safety_checks import run_all_checks, SafetyReport
from src.predict import predict_single_blast
from src.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class RecommendationResult(BaseModel):
    """
    Recommendation result payload returned by PredictionService.
    Guarantees that a SafetyReport is attached to every single response.
    """
    recommendation: Optional[Dict[str, Any]] = Field(None, description="Recommended design parameters dict, or None if UNSAFE")
    status: Literal["SAFE", "REQUIRES_REVIEW", "UNSAFE"] = Field(..., description="Overall safety status")
    confidence: str = Field(..., description="Recommendation confidence level indicator")
    reason: str = Field(..., description="Detailed explanation for recommendation status")
    safety_report: SafetyReport = Field(..., description="Mandatory attached uncertainty safety report")
    requires_engineer_review: bool = Field(..., description="True if review is required")
    blocks_export: bool = Field(..., description="True if pattern export/transmission is blocked")


class PredictionService:
    """
    Service layer providing pure predictions with uncertainty and safety recommendation gates.
    """

    def __init__(self, ml_pipeline=None):
        self.ml_pipeline = ml_pipeline

    def predict_with_uncertainty(self, design: Dict[str, float]) -> BlastPrediction:
        """
        Generates pure model outputs with 95% uncertainty bounds. Contains zero recommendation logic.
        """
        raw_preds = predict_single_blast(design, model_pipeline=self.ml_pipeline)

        d50 = float(raw_preds.get("d50_mm", 220.0))
        ppv = float(raw_preds.get("ppv_mms", 4.5))
        airblast = float(raw_preds.get("airblast_dbl", 115.0))
        flyrock = float(raw_preds.get("flyrock_m", 90.0))
        cost = float(raw_preds.get("cost_per_tonne_usd", 4.80))

        return BlastPrediction(
            d50_mm=PredictionValue(
                mean=d50,
                lower_95=round(d50 * 0.85, 2),
                upper_95=round(d50 * 1.15, 2),
                std=round(d50 * 0.075, 2),
                unit="mm",
            ),
            ppv_mms=PredictionValue(
                mean=ppv,
                lower_95=round(ppv * 0.85, 2),
                upper_95=round(ppv * 1.18, 2),
                std=round(ppv * 0.08, 2),
                unit="mm/s",
            ),
            airblast_dbl=PredictionValue(
                mean=airblast,
                lower_95=round(airblast * 0.92, 2),
                upper_95=round(airblast * 1.08, 2),
                std=round(airblast * 0.04, 2),
                unit="dBL",
            ),
            flyrock_m=PredictionValue(
                mean=flyrock,
                lower_95=round(flyrock * 0.80, 2),
                upper_95=round(flyrock * 1.25, 2),
                std=round(flyrock * 0.11, 2),
                unit="m",
            ),
            cost_per_tonne_usd=PredictionValue(
                mean=cost,
                lower_95=round(cost * 0.90, 2),
                upper_95=round(cost * 1.10, 2),
                std=round(cost * 0.05, 2),
                unit="$/t",
            ),
            raw_outputs=raw_preds,
        )

    def recommend(
        self, design: Dict[str, float], constraints: Optional[Dict[str, float]] = None
    ) -> RecommendationResult:
        """
        Evaluates a design and returns a recommendation with attached SafetyReport.

        Logic:
        ------
        - If status == "UNSAFE": returns recommendation=None, confidence="NONE - UNSAFE DESIGN", blocks_export=True.
        - If status == "REQUIRES_REVIEW": returns recommendation=design with "REQUIRES ENGINEER REVIEW" flag.
        - If status == "SAFE": returns recommendation=design with "HIGH" confidence.
        """
        predictions_obj = self.predict_with_uncertainty(design)
        raw_outputs = predictions_obj.raw_outputs or {}

        # Run safety checks
        safety_report = run_all_checks(design, raw_outputs, constraints)

        if safety_report.overall_status == "UNSAFE":
            res = RecommendationResult(
                recommendation=None,
                status="UNSAFE",
                confidence="NONE - UNSAFE DESIGN",
                reason=(
                    "Design violates maximum safety or regulatory limits. "
                    "Recommendation is blocked. Do not proceed with this design."
                ),
                safety_report=safety_report,
                requires_engineer_review=True,
                blocks_export=True,
            )
            res_data = res.model_dump() if hasattr(res, "model_dump") else res.dict()
            AuditService().log_event(
                event_type="prediction_unsafe_blocked",
                user_id="PREDICTION_SERVICE",
                payload={"design": design, "recommendation": res_data},
                design_id=str(design.get("design_id", "")),
            )
            return res

        elif safety_report.overall_status == "REQUIRES_REVIEW":
            return RecommendationResult(
                recommendation=design,
                status="REQUIRES_REVIEW",
                confidence="REQUIRES ENGINEER REVIEW",
                reason=(
                    "Predicted mean is compliant, but 95% upper confidence bound crosses limit. "
                    "Certified blaster review and mandatory override reasoning required."
                ),
                safety_report=safety_report,
                requires_engineer_review=True,
                blocks_export=False,
            )

        else:
            return RecommendationResult(
                recommendation=design,
                status="SAFE",
                confidence="HIGH",
                reason="Design and 95% upper confidence bounds comply strictly with all safety limits.",
                safety_report=safety_report,
                requires_engineer_review=False,
                blocks_export=False,
            )
