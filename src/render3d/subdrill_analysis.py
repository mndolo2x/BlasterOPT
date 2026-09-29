"""
Subdrill depth optimization and analysis for BlasterOPT 3D engine.
"""

from typing import Dict, Any, Tuple, Optional


def calculate_optimal_subdrill(
    burden_m: float,
    bench_height_m: float,
    rock_factor_a: float,
    hole_diameter_mm: float,
) -> Dict[str, Any]:
    """
    Calculate the optimal subdrill depth below the bench floor.

    Industry rule of thumb (Konya, 1995, "Blast Design"):
        subdrill = 0.3 * burden

    Adjusted by rock factor:
        subdrill = 0.3 * burden * (1 + (rock_factor_a - 8) * 0.05)

    Clamped to [0.1 * bench_height, 0.5 * bench_height].

    Returns:
        {
            "optimal_subdrill_m": float,
            "recommended_range_m": (float, float),
            "reasoning": str,
            "excessive_risk": str | None,
            "insufficient_risk": str | None,
        }
    """
    if burden_m <= 0:
        raise ValueError(f"burden_m ({burden_m}) must be positive")
    if bench_height_m <= 0:
        raise ValueError(f"bench_height_m ({bench_height_m}) must be positive")

    # Formula: subdrill = 0.3 * burden * (1 + (rock_factor_a - 8) * 0.05)
    base_subdrill = 0.3 * burden_m
    rock_adj = 1.0 + (rock_factor_a - 8.0) * 0.05
    raw_subdrill = base_subdrill * rock_adj

    # Clamped to [0.05 * bench_height, 0.5 * bench_height]
    min_limit = 0.05 * bench_height_m
    max_limit = 0.5 * bench_height_m
    optimal_subdrill = float(min(max(raw_subdrill, min_limit), max_limit))

    rec_min = float(round(optimal_subdrill * 0.85, 2))
    rec_max = float(round(optimal_subdrill * 1.15, 2))

    reasoning = (
        f"Base Konya subdrill (0.3 * {burden_m}m = {base_subdrill:.2f}m) "
        f"adjusted for rock factor A={rock_factor_a} to {raw_subdrill:.2f}m, "
        f"clamped within [{min_limit:.2f}m, {max_limit:.2f}m]."
    )

    excessive_risk: Optional[str] = None
    if optimal_subdrill > 0.35 * bench_height_m:
        excessive_risk = "Excessive subdrilling risk: increased floor damage, grade dilution, and drilling energy waste."

    insufficient_risk: Optional[str] = None
    if optimal_subdrill < 0.15 * bench_height_m:
        insufficient_risk = "Insufficient subdrilling risk: high toe formation, tight floor, and expensive mechanical secondary breaking."

    return {
        "optimal_subdrill_m": round(optimal_subdrill, 2),
        "recommended_range_m": (rec_min, rec_max),
        "reasoning": reasoning,
        "excessive_risk": excessive_risk,
        "insufficient_risk": insufficient_risk,
    }


def analyze_subdrill_existing(
    current_subdrill_m: float, optimal_m: float
) -> Dict[str, Any]:
    """
    Compare current subdrill to optimal.
    Returns classification: 'optimal', 'excessive', 'insufficient'.
    """
    if optimal_m <= 0:
        raise ValueError(f"optimal_m ({optimal_m}) must be positive")

    diff = current_subdrill_m - optimal_m
    pct_diff = (diff / optimal_m) * 100.0

    if abs(pct_diff) <= 15.0:
        classification = "optimal"
        status_msg = "Current subdrill is within optimal bounds."
    elif diff > 0:
        classification = "excessive"
        status_msg = f"Current subdrill is excessive by {diff:.2f}m ({pct_diff:.1f}%)."
    else:
        classification = "insufficient"
        status_msg = f"Current subdrill is insufficient by {abs(diff):.2f}m ({abs(pct_diff):.1f}%)."

    return {
        "current_subdrill_m": round(current_subdrill_m, 2),
        "optimal_subdrill_m": round(optimal_m, 2),
        "difference_m": round(diff, 2),
        "percentage_difference": round(pct_diff, 1),
        "classification": classification,
        "status_message": status_msg,
    }
