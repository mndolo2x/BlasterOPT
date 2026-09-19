"""
Uncertainty-Aware Safety Checks Module for BlastOpt Botswana.

Evaluates predicted blast outcomes (PPV, airblast overpressure, flyrock range, stemming confinement)
against site and regulatory limits using 95% confidence intervals to prevent unsafe operations
near regulatory thresholds.
"""

import os
import logging
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple, Literal

logger = logging.getLogger(__name__)


class SafetyCheck(BaseModel):
    """
    Individual safety check model for a specific predicted parameter.
    """
    check_name: str = Field(..., description="Name of parameter checked (e.g. ppv_limit, airblast_limit)")
    predicted_value: float = Field(..., description="Mean predicted value")
    lower_95: float = Field(..., description="Lower 95% confidence bound")
    upper_95: float = Field(..., description="Upper 95% confidence bound")
    limit: float = Field(..., description="Regulatory or site threshold limit")
    status: Literal["SAFE", "REQUIRES_REVIEW", "UNSAFE"] = Field(..., description="Safety evaluation status")
    reasoning: str = Field(..., description="Explanation of safety evaluation")


class SafetyReport(BaseModel):
    """
    Aggregated safety report containing all parameter checks and overall compliance state.
    """
    overall_status: Literal["SAFE", "REQUIRES_REVIEW", "UNSAFE"] = Field(..., description="Overall blast safety status")
    checks: List[SafetyCheck] = Field(default_factory=list, description="List of individual parameter checks")
    requires_engineer_review: bool = Field(..., description="True if review or override is required")
    blocks_export: bool = Field(..., description="True if design export/transmission must be blocked")


def evaluate_safety(
    predictions: Dict[str, float],
    limits: Optional[Dict[str, float]] = None,
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    blast_params: Optional[Dict[str, float]] = None,
) -> SafetyReport:
    """
    Evaluates blast design predictions against limits incorporating 95% uncertainty bounds.

    Rules:
    ------
    - "UNSAFE": Predicted mean exceeds maximum limit (or falls below minimum limit).
    - "REQUIRES_REVIEW": Predicted mean is within limit, but 95% confidence bound crosses limit.
    - "SAFE": Upper 95% confidence bound is strictly within limit.

    Parameters:
    -----------
    predictions : Dict[str, float]
        Predicted outcomes (ppv_mms, airblast_dbl, flyrock_m, d50_mm).
    limits : Dict[str, float], optional
        Active regulatory or site limits.
    confidence_intervals : Dict[str, Tuple[float, float]], optional
        Optional explicit 95% confidence bounds dict {"param": (lower_95, upper_95)}.
    blast_params : Dict[str, float], optional
        Input design parameters (stemming_m, powder_factor_kg_m3).

    Returns:
    --------
    SafetyReport
        Pydantic SafetyReport with overall status and individual checks.
    """
    if limits is None:
        limits = {
            "max_ppv_mms": 10.0,
            "max_airblast_dbl": 120.0,
            "max_flyrock_m": 250.0,
            "min_stemming_m": 2.5,
            "max_powder_factor_kg_m3": 1.20,
        }

    confidence_intervals = confidence_intervals or {}
    blast_params = blast_params or {}

    checks: List[SafetyCheck] = []

    # 1. Ground Vibration (PPV)
    if "ppv_mms" in predictions or "pred_ppv_mms" in predictions:
        pred_ppv = float(predictions.get("ppv_mms", predictions.get("pred_ppv_mms", 0.0)))
        limit_ppv = float(limits.get("max_ppv_mms", 10.0))
        ci = confidence_intervals.get("ppv_mms", (pred_ppv * 0.85, pred_ppv * 1.18))
        lower_95, upper_95 = float(ci[0]), float(ci[1])

        if pred_ppv > limit_ppv:
            status = "UNSAFE"
            reasoning = f"Mean predicted PPV ({pred_ppv:.2f} mm/s) exceeds maximum limit ({limit_ppv:.1f} mm/s)."
        elif upper_95 > limit_ppv:
            status = "REQUIRES_REVIEW"
            reasoning = (
                f"Mean predicted PPV ({pred_ppv:.2f} mm/s) is within limit ({limit_ppv:.1f} mm/s), "
                f"but 95% upper confidence bound ({upper_95:.2f} mm/s) exceeds limit."
            )
        else:
            status = "SAFE"
            reasoning = f"Predicted PPV ({pred_ppv:.2f} mm/s, U95={upper_95:.2f} mm/s) is within safe limit ({limit_ppv:.1f} mm/s)."

        checks.append(SafetyCheck(
            check_name="ppv_limit",
            predicted_value=pred_ppv,
            lower_95=lower_95,
            upper_95=upper_95,
            limit=limit_ppv,
            status=status,
            reasoning=reasoning,
        ))

    # 2. Airblast Overpressure (dBL)
    if "airblast_dbl" in predictions or "pred_airblast_dbl" in predictions:
        pred_ab = float(predictions.get("airblast_dbl", predictions.get("pred_airblast_dbl", 0.0)))
        limit_ab = float(limits.get("max_airblast_dbl", 120.0))
        ci = confidence_intervals.get("airblast_dbl", (pred_ab * 0.92, pred_ab * 1.08))
        lower_95, upper_95 = float(ci[0]), float(ci[1])

        if pred_ab > limit_ab:
            status = "UNSAFE"
            reasoning = f"Mean predicted Airblast ({pred_ab:.1f} dBL) exceeds limit ({limit_ab:.1f} dBL)."
        elif upper_95 > limit_ab:
            status = "REQUIRES_REVIEW"
            reasoning = (
                f"Mean predicted Airblast ({pred_ab:.1f} dBL) is within limit ({limit_ab:.1f} dBL), "
                f"but 95% upper bound ({upper_95:.1f} dBL) exceeds limit."
            )
        else:
            status = "SAFE"
            reasoning = f"Predicted Airblast ({pred_ab:.1f} dBL, U95={upper_95:.1f} dBL) is safe."

        checks.append(SafetyCheck(
            check_name="airblast_limit",
            predicted_value=pred_ab,
            lower_95=lower_95,
            upper_95=upper_95,
            limit=limit_ab,
            status=status,
            reasoning=reasoning,
        ))

    # 3. Flyrock Range
    if "flyrock_m" in predictions or "pred_flyrock_m" in predictions:
        pred_fly = float(predictions.get("flyrock_m", predictions.get("pred_flyrock_m", 0.0)))
        limit_fly = float(limits.get("max_flyrock_m", 250.0))
        ci = confidence_intervals.get("flyrock_m", (pred_fly * 0.80, pred_fly * 1.25))
        lower_95, upper_95 = float(ci[0]), float(ci[1])

        if pred_fly > limit_fly:
            status = "UNSAFE"
            reasoning = f"Mean predicted Flyrock ({pred_fly:.1f} m) exceeds boundary limit ({limit_fly:.1f} m)."
        elif upper_95 > limit_fly:
            status = "REQUIRES_REVIEW"
            reasoning = (
                f"Mean predicted Flyrock ({pred_fly:.1f} m) is within limit ({limit_fly:.1f} m), "
                f"but 95% upper bound ({upper_95:.1f} m) crosses safety threshold."
            )
        else:
            status = "SAFE"
            reasoning = f"Predicted Flyrock ({pred_fly:.1f} m, U95={upper_95:.1f} m) is safe."

        checks.append(SafetyCheck(
            check_name="flyrock_limit",
            predicted_value=pred_fly,
            lower_95=lower_95,
            upper_95=upper_95,
            limit=limit_fly,
            status=status,
            reasoning=reasoning,
        ))

    # 4. Stemming Confinement Check
    if "stemming_m" in blast_params:
        stemming_val = float(blast_params["stemming_m"])
        limit_stem = float(limits.get("min_stemming_m", 2.5))
        ci = confidence_intervals.get("stemming_m", (stemming_val * 0.95, stemming_val * 1.05))
        lower_95, upper_95 = float(ci[0]), float(ci[1])

        if stemming_val < limit_stem:
            status = "UNSAFE"
            reasoning = f"Stemming length ({stemming_val:.2f} m) is below minimum confinement limit ({limit_stem:.1f} m)."
        elif lower_95 < limit_stem:
            status = "REQUIRES_REVIEW"
            reasoning = (
                f"Stemming length ({stemming_val:.2f} m) meets limit ({limit_stem:.1f} m), "
                f"but lower 95% tolerance bound ({lower_95:.2f} m) risks blowout."
            )
        else:
            status = "SAFE"
            reasoning = f"Stemming confinement ({stemming_val:.2f} m) is safe."

        checks.append(SafetyCheck(
            check_name="stemming_confinement",
            predicted_value=stemming_val,
            lower_95=lower_95,
            upper_95=upper_95,
            limit=limit_stem,
            status=status,
            reasoning=reasoning,
        ))

    # Determine overall status
    statuses = [c.status for c in checks]
    if "UNSAFE" in statuses:
        overall_status = "UNSAFE"
        requires_review = True
        blocks_export = True
    elif "REQUIRES_REVIEW" in statuses:
        overall_status = "REQUIRES_REVIEW"
        requires_review = True
        blocks_export = False
    else:
        overall_status = "SAFE"
        requires_review = False
        blocks_export = False

    return SafetyReport(
        overall_status=overall_status,
        checks=checks,
        requires_engineer_review=requires_review,
        blocks_export=blocks_export,
    )
