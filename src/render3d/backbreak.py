"""
Backbreak prediction and prevention analysis for BlasterOPT 3D engine.
"""

from typing import Dict, Any, List


def predict_backbreak(
    powder_factor_kg_m3: float,
    stemming_m: float,
    burden_m: float,
    rock_factor_a: float,
    joint_spacing_m: float,
) -> Dict[str, Any]:
    """
    Predict backbreak behind the last row of holes.

    Empirical formula (Scoble et al., 1997):
        backbreak_m = 2.5 * (powder_factor_kg_m3 ** 0.5)
                       * (1 - stemming_m / burden_m) ** 0.3
                       * (rock_factor_a / 8) ** 0.4
                       * (1 + 1 / (joint_spacing_m + 0.5))

    Returns:
        {
            "predicted_backbreak_m": float,
            "risk_level": str,  # 'low', 'medium', 'high'
            "contributing_factors": list[str],
            "prevention_measures": list[str],
        }
    """
    if powder_factor_kg_m3 <= 0:
        raise ValueError(f"powder_factor_kg_m3 ({powder_factor_kg_m3}) must be positive")
    if burden_m <= 0:
        raise ValueError(f"burden_m ({burden_m}) must be positive")
    if joint_spacing_m < 0:
        raise ValueError(f"joint_spacing_m ({joint_spacing_m}) cannot be negative")

    # Formula (Scoble et al., 1997):
    # backbreak_m = 2.5 * (powder_factor_kg_m3 ** 0.5) * (1 - stemming_m / burden_m) ** 0.3 * (rock_factor_a / 8) ** 0.4 * (1 + 1 / (joint_spacing_m + 0.5))
    stemming_ratio = min(0.95, max(0.0, stemming_m / burden_m))
    term1 = 2.5 * (powder_factor_kg_m3 ** 0.5)
    term2 = max(0.01, 1.0 - stemming_ratio) ** 0.3
    term3 = (max(0.1, rock_factor_a) / 8.0) ** 0.4
    term4 = 1.0 + 1.0 / (joint_spacing_m + 0.5)

    backbreak_m = float(term1 * term2 * term3 * term4)

    factors: List[str] = []
    if powder_factor_kg_m3 > 0.8:
        factors.append(f"High powder factor ({powder_factor_kg_m3:.2f} kg/m³)")
    if stemming_ratio < 0.7:
        factors.append(f"Short stemming length relative to burden ({stemming_m:.2f}m vs {burden_m:.2f}m burden)")
    if rock_factor_a > 10.0:
        factors.append(f"Hard / massive rock mass rating (Rock factor A={rock_factor_a:.1f})")
    if joint_spacing_m < 0.5:
        factors.append(f"Closely spaced joints / fractured rock ({joint_spacing_m:.2f}m joint spacing)")

    if backbreak_m < 1.5:
        risk_level = "low"
    elif backbreak_m <= 3.0:
        risk_level = "medium"
    else:
        risk_level = "high"

    prevention = recommend_backbreak_prevention(backbreak_m, bench_critical=(risk_level == "high"))

    return {
        "predicted_backbreak_m": round(backbreak_m, 2),
        "risk_level": risk_level,
        "contributing_factors": factors,
        "prevention_measures": prevention,
    }


def recommend_backbreak_prevention(backbreak_m: float, bench_critical: bool) -> List[str]:
    """
    Returns a prioritized list of prevention measures:
    - Increase stemming length
    - Reduce powder factor in back row
    - Use decoupled charges
    - Reduce inter-row delay
    - Add a buffer row
    """
    measures: List[str] = []

    if backbreak_m > 1.0 or bench_critical:
        measures.append("Increase stemming length on back row by 0.5 - 1.0 m to contain explosive energy")
        measures.append("Reduce powder factor in back row using deck charges or lighter explosive density")
        measures.append("Use decoupled or trimmed explosive charges along crest line")
        measures.append("Reduce inter-row delay interval to minimize crest shock wave reflection")
        measures.append("Add a uncharged/lightly charged buffer row between production holes and final wall")

    return measures
