"""
GSI Correction Module for Site-Specific Vibration Attenuation.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from src.site_calibration.models import AttenuationParameters, PPVPrediction

logger = logging.getLogger(__name__)


def calculate_gsi_factor(gsi: float, a: float = 1.0, b: float = 0.0) -> float:
    """
    Computes GSI correction multiplier f(GSI) = a * exp(b * GSI).
    """
    if a is None or b is None:
        return 1.0
    return float(a * np.exp(b * float(gsi)))


def predict_gsi_modified_ppv(
    distance_m: float,
    charge_per_delay_kg: float,
    params: AttenuationParameters,
    gsi: Optional[float] = None,
) -> PPVPrediction:
    """
    Predicts PPV using GSI-modified USBM equation:
    PPV = K * (D / sqrt(Q))^(-B) * a * exp(b * GSI)

    Falls back to standard USBM if GSI or GSI correction parameters are unavailable.
    """
    sd = distance_m / np.sqrt(charge_per_delay_kg)
    k_val = params.K
    b_val = params.B

    # Standard USBM PPV base prediction
    base_ppv = k_val * (sd ** (-b_val))

    # Check GSI availability
    if gsi is not None and params.gsi_a is not None and params.gsi_b is not None and params.gsi_b != 0.0:
        gsi_factor = calculate_gsi_factor(gsi, params.gsi_a, params.gsi_b)
        predicted_ppv = float(base_ppv * gsi_factor)
        method = "GSI_MODIFIED_USBM"
        params_used = {
            "K": k_val,
            "B": b_val,
            "gsi_a": params.gsi_a,
            "gsi_b": params.gsi_b,
            "gsi_value": gsi,
            "gsi_factor": gsi_factor,
        }
    else:
        predicted_ppv = float(base_ppv)
        method = "USBM_SITE_CALIBRATED"
        params_used = {"K": k_val, "B": b_val}

    # 95% Prediction Confidence Bounds (+/- 25% relative margin based on RMSE)
    rel_margin = min(0.35, max(0.15, params.rmse / max(1.0, predicted_ppv)))
    lower_95 = float(max(0.01, predicted_ppv * (1.0 - rel_margin)))
    upper_95 = float(predicted_ppv * (1.0 + rel_margin))

    return PPVPrediction(
        ppv_mm_s=round(predicted_ppv, 2),
        lower_95=round(lower_95, 2),
        upper_95=round(upper_95, 2),
        method=method,
        parameters_used=params_used,
    )


def evaluate_gsi_improvement(
    r2_standard: float,
    r2_gsi_modified: float
) -> Dict[str, Any]:
    """
    Compares R2 scores with and without GSI correction to evaluate improvement percentage.
    """
    improvement_pct = max(0.0, ((r2_gsi_modified - r2_standard) / max(0.01, r2_standard)) * 100.0)
    is_significant = improvement_pct >= 5.0

    return {
        "r2_standard": round(r2_standard, 4),
        "r2_gsi_modified": round(r2_gsi_modified, 4),
        "improvement_pct": round(improvement_pct, 2),
        "is_significant": is_significant,
        "recommendation": "Adopt GSI-modified attenuation model." if is_significant else "Use standard USBM model."
    }
