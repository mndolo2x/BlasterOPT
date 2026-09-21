"""
Uncertainty-Aware Safety Checks Module for BlastOpt Botswana.

Evaluates predicted blast outcomes (PPV, airblast overpressure, flyrock range, stemming confinement)
against site and regulatory limits using 95% confidence intervals to prevent unsafe operations
near regulatory thresholds.
"""

import os
import logging
from pydantic import BaseModel, Field
try:
    from typing import Dict, Any, List, Optional, Tuple, Literal, Union
except ImportError:
    from typing import Dict, Any, List, Optional, Tuple, Union
    from typing_extensions import Literal

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


def check_ppv(
    ppv: float, max_ppv: float, lower_95: Optional[float] = None, upper_95: Optional[float] = None
) -> SafetyCheck:
    """Individual Ground Vibration (PPV) safety check."""
    l95 = lower_95 if lower_95 is not None else ppv * 0.85
    u95 = upper_95 if upper_95 is not None else ppv * 1.18

    if ppv > max_ppv:
        status = "UNSAFE"
        reasoning = f"Mean predicted PPV ({ppv:.2f} mm/s) exceeds maximum limit ({max_ppv:.1f} mm/s)."
    elif u95 > max_ppv:
        status = "REQUIRES_REVIEW"
        reasoning = (
            f"Mean predicted PPV ({ppv:.2f} mm/s) is within limit ({max_ppv:.1f} mm/s), "
            f"but 95% upper confidence bound ({u95:.2f} mm/s) exceeds limit."
        )
    else:
        status = "SAFE"
        reasoning = f"Predicted PPV ({ppv:.2f} mm/s, U95={u95:.2f} mm/s) is within safe limit ({max_ppv:.1f} mm/s)."

    return SafetyCheck(
        check_name="ppv_limit",
        predicted_value=ppv,
        lower_95=l95,
        upper_95=u95,
        limit=max_ppv,
        status=status,
        reasoning=reasoning,
    )


def check_airblast(
    airblast: float, max_airblast: float, lower_95: Optional[float] = None, upper_95: Optional[float] = None
) -> SafetyCheck:
    """Individual Airblast Overpressure safety check."""
    l95 = lower_95 if lower_95 is not None else airblast * 0.92
    u95 = upper_95 if upper_95 is not None else airblast * 1.08

    if airblast > max_airblast:
        status = "UNSAFE"
        reasoning = f"Mean predicted Airblast ({airblast:.1f} dBL) exceeds limit ({max_airblast:.1f} dBL)."
    elif u95 > max_airblast:
        status = "REQUIRES_REVIEW"
        reasoning = (
            f"Mean predicted Airblast ({airblast:.1f} dBL) is within limit ({max_airblast:.1f} dBL), "
            f"but 95% upper bound ({u95:.1f} dBL) exceeds limit."
        )
    else:
        status = "SAFE"
        reasoning = f"Predicted Airblast ({airblast:.1f} dBL, U95={u95:.1f} dBL) is safe."

    return SafetyCheck(
        check_name="airblast_limit",
        predicted_value=airblast,
        lower_95=l95,
        upper_95=u95,
        limit=max_airblast,
        status=status,
        reasoning=reasoning,
    )


def check_flyrock(
    flyrock: float, max_flyrock: float, lower_95: Optional[float] = None, upper_95: Optional[float] = None
) -> SafetyCheck:
    """Individual Flyrock distance safety check."""
    l95 = lower_95 if lower_95 is not None else flyrock * 0.80
    u95 = upper_95 if upper_95 is not None else flyrock * 1.25

    if flyrock > max_flyrock:
        status = "UNSAFE"
        reasoning = f"Mean predicted Flyrock ({flyrock:.1f} m) exceeds boundary limit ({max_flyrock:.1f} m)."
    elif u95 > max_flyrock:
        status = "REQUIRES_REVIEW"
        reasoning = (
            f"Mean predicted Flyrock ({flyrock:.1f} m) is within limit ({max_flyrock:.1f} m), "
            f"but 95% upper bound ({u95:.1f} m) crosses safety threshold."
        )
    else:
        status = "SAFE"
        reasoning = f"Predicted Flyrock ({flyrock:.1f} m, U95={u95:.1f} m) is safe."

    return SafetyCheck(
        check_name="flyrock_limit",
        predicted_value=flyrock,
        lower_95=l95,
        upper_95=u95,
        limit=max_flyrock,
        status=status,
        reasoning=reasoning,
    )


def determine_overall_status(checks: List[SafetyCheck]) -> Literal["SAFE", "REQUIRES_REVIEW", "UNSAFE"]:
    """Determines overall aggregated status from list of SafetyCheck items."""
    statuses = [c.status for c in checks]
    if "UNSAFE" in statuses:
        return "UNSAFE"
    elif "REQUIRES_REVIEW" in statuses:
        return "REQUIRES_REVIEW"
    return "SAFE"


def run_all_checks(design: Any, predictions: Any, constraints: Any) -> SafetyReport:
    """
    Runs all safety checks for a given blast design, predictions, and constraints.
    """
    def _get_val(obj: Any, keys: List[str], default: float) -> float:
        if isinstance(obj, dict):
            for k in keys:
                if k in obj and obj[k] is not None:
                    return float(obj[k])
        else:
            for k in keys:
                if hasattr(obj, k) and getattr(obj, k) is not None:
                    return float(getattr(obj, k))
        return default

    pred_ppv = _get_val(predictions, ["ppv", "ppv_mms", "pred_ppv_mms"], 5.0)
    pred_airblast = _get_val(predictions, ["airblast", "airblast_dbl", "pred_airblast_dbl"], 110.0)
    pred_flyrock = _get_val(predictions, ["flyrock", "flyrock_m", "pred_flyrock_m"], 100.0)

    max_ppv = _get_val(constraints, ["max_ppv_mm_s", "max_ppv_mms", "max_ppv"], 10.0)
    max_airblast = _get_val(constraints, ["max_airblast_db", "max_airblast_dbl", "max_airblast"], 120.0)
    max_flyrock = _get_val(constraints, ["flyrock_exclusion_zone_m", "max_flyrock_m", "max_flyrock"], 250.0)

    checks = [
        check_ppv(pred_ppv, max_ppv),
        check_airblast(pred_airblast, max_airblast),
        check_flyrock(pred_flyrock, max_flyrock),
    ]

    overall = determine_overall_status(checks)
    return SafetyReport(
        overall_status=overall,
        checks=checks,
        requires_engineer_review=any(c.status in ["REQUIRES_REVIEW", "UNSAFE"] for c in checks),
        blocks_export=any(c.status == "UNSAFE" for c in checks),
    )


def evaluate_safety(
    predictions: Dict[str, float],
    limits: Optional[Dict[str, float]] = None,
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    blast_params: Optional[Dict[str, float]] = None,
) -> SafetyReport:
    """
    Evaluates blast design predictions against limits incorporating 95% uncertainty bounds.
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
    if "ppv_mms" in predictions or "pred_ppv_mms" in predictions or "ppv" in predictions:
        pred_ppv = float(predictions.get("ppv_mms", predictions.get("pred_ppv_mms", predictions.get("ppv", 0.0))))
        limit_ppv = float(limits.get("max_ppv_mms", limits.get("max_ppv_mm_s", limits.get("max_ppv", 10.0))))
        ci = confidence_intervals.get("ppv_mms", confidence_intervals.get("ppv", (pred_ppv * 0.85, pred_ppv * 1.18)))
        checks.append(check_ppv(pred_ppv, limit_ppv, lower_95=ci[0], upper_95=ci[1]))

    # 2. Airblast Overpressure (dBL)
    if "airblast_dbl" in predictions or "pred_airblast_dbl" in predictions or "airblast" in predictions:
        pred_ab = float(predictions.get("airblast_dbl", predictions.get("pred_airblast_dbl", predictions.get("airblast", 0.0))))
        limit_ab = float(limits.get("max_airblast_dbl", limits.get("max_airblast_db", limits.get("max_airblast", 120.0))))
        ci = confidence_intervals.get("airblast_dbl", confidence_intervals.get("airblast", (pred_ab * 0.92, pred_ab * 1.08)))
        checks.append(check_airblast(pred_ab, limit_ab, lower_95=ci[0], upper_95=ci[1]))

    # 3. Flyrock Range
    if "flyrock_m" in predictions or "pred_flyrock_m" in predictions or "flyrock" in predictions:
        pred_fly = float(predictions.get("flyrock_m", predictions.get("pred_flyrock_m", predictions.get("flyrock", 0.0))))
        limit_fly = float(limits.get("max_flyrock_m", limits.get("flyrock_exclusion_zone_m", limits.get("max_flyrock", 250.0))))
        ci = confidence_intervals.get("flyrock_m", confidence_intervals.get("flyrock", (pred_fly * 0.80, pred_fly * 1.25)))
        checks.append(check_flyrock(pred_fly, limit_fly, lower_95=ci[0], upper_95=ci[1]))

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

    overall = determine_overall_status(checks)
    return SafetyReport(
        overall_status=overall,
        checks=checks,
        requires_engineer_review=any(c.status in ["REQUIRES_REVIEW", "UNSAFE"] for c in checks),
        blocks_export=any(c.status == "UNSAFE" for c in checks),
    )
