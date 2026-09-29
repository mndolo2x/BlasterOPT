"""
Toe burden analysis and relief hole calculation for BlasterOPT 3D engine.
"""

import math
from typing import Dict, Any, List


def calculate_toe_burden(
    crest_burden_m: float, bench_height_m: float, face_angle_deg: float
) -> Dict[str, Any]:
    """
    Calculate the toe burden from the crest burden and bench geometry.

    Formula (Atlas Copco, "Blasting Manual"):
        toe_burden = crest_burden + (bench_height / tan(face_angle_radians))

    Example: crest_burden=4.0, bench_height=15.0, face_angle=75°
        toe_burden = 4.0 + (15.0 / tan(75°)) = 4.0 + 4.02 = 8.02 m

    Returns:
        {
            "crest_burden_m": float,
            "toe_burden_m": float,
            "difference_m": float,
            "requires_relief": bool,
            "recommendation": str,
        }
    """
    if crest_burden_m <= 0:
        raise ValueError(f"crest_burden_m ({crest_burden_m}) must be positive")
    if bench_height_m <= 0:
        raise ValueError(f"bench_height_m ({bench_height_m}) must be positive")
    if not (10.0 <= face_angle_deg <= 90.0):
        raise ValueError(f"face_angle_deg ({face_angle_deg}) must be between 10 and 90 degrees")

    # Formula: toe_burden = crest_burden + (bench_height / tan(face_angle_radians))
    face_rad = math.radians(face_angle_deg)
    horizontal_offset = bench_height_m / math.tan(face_rad)
    toe_burden = crest_burden_m + horizontal_offset
    diff = toe_burden - crest_burden_m

    # Toe burden exceeds 1.4 * crest_burden requires relief holes
    requires_relief = toe_burden > (1.4 * crest_burden_m)

    if requires_relief:
        rec = f"Toe burden ({toe_burden:.2f}m) significantly exceeds crest burden ({crest_burden_m:.2f}m). Relief holes or angled drilling recommended."
    else:
        rec = f"Toe burden ({toe_burden:.2f}m) is acceptable relative to crest burden ({crest_burden_m:.2f}m)."

    return {
        "crest_burden_m": round(crest_burden_m, 2),
        "toe_burden_m": round(toe_burden, 2),
        "difference_m": round(diff, 2),
        "requires_relief": requires_relief,
        "recommendation": rec,
    }


def calculate_relief_holes(
    toe_burden_m: float, target_burden_m: float, spacing_m: float
) -> Dict[str, Any]:
    """
    When toe burden exceeds target burden, relief holes are needed
    to prevent hard toe and poor floor conditions.

    Returns the number and positions of relief holes required.
    """
    if target_burden_m <= 0:
        raise ValueError(f"target_burden_m ({target_burden_m}) must be positive")

    excess_burden = max(0.0, toe_burden_m - target_burden_m)
    if excess_burden <= 0.1:
        return {
            "relief_holes_required": 0,
            "positions_m": [],
            "reasoning": "Toe burden is within target burden limits.",
        }

    # Estimate required relief holes per hole interval along toe
    num_relief = int(math.ceil(excess_burden / (target_burden_m * 0.75)))
    positions = []
    step = spacing_m / (num_relief + 1)
    for i in range(1, num_relief + 1):
        positions.append(round(i * step, 2))

    return {
        "relief_holes_required": num_relief,
        "positions_m": positions,
        "reasoning": f"Excess toe burden of {excess_burden:.2f}m requires {num_relief} relief hole(s) spaced along face interval.",
    }
