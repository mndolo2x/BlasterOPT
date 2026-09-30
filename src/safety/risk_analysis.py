"""
Comprehensive quantitative risk matrix per ISO 31000 / AS/NZS 4360 risk management standard.
"""

from typing import Dict, Any, List


def calculate_risk_matrix(
    ppv_mms: float,
    airblast_dbl: float,
    flyrock_m: float,
    hole_collisions_count: int,
    backbreak_m: float,
) -> Dict[str, Any]:
    """
    Calculate ISO 31000 quantitative risk matrix score and hazard ranking for blast design.

    Returns:
        {
            "overall_risk_score": float,  # 1 to 25
            "risk_level": str,  # 'low', 'medium', 'high', 'extreme'
            "hazards": list[dict],
            "action_plan": list[str],
        }
    """
    hazards: List[Dict[str, Any]] = []
    actions: List[str] = []

    # 1. Vibration
    if ppv_mms > 10.0:
        hazards.append({"hazard": "Ground Vibration Exceedance", "severity": 5, "likelihood": 4, "score": 20, "level": "high"})
        actions.append("Reduce max charge per delay or increase monitoring distance.")

    # 2. Flyrock
    if flyrock_m > 250.0:
        hazards.append({"hazard": "Flyrock Exclusion Zone Breach", "severity": 5, "likelihood": 5, "score": 25, "level": "extreme"})
        actions.append("Increase stemming height and check front row burden.")

    # 3. Collision
    if hole_collisions_count > 0:
        hazards.append({"hazard": "Hole-to-Hole Toe Collision", "severity": 5, "likelihood": 4, "score": 20, "level": "high"})
        actions.append("Re-design collar spacing or adjust drill hole dip angles.")

    # 4. Airblast
    if airblast_dbl > 120.0:
        hazards.append({"hazard": "Airblast Overpressure Exceedance", "severity": 4, "likelihood": 4, "score": 16, "level": "medium"})
        actions.append("Cover surface detonation trunklines and avoid blasting under temperature inversions.")

    max_score = max([h["score"] for h in hazards], default=4)

    if max_score >= 20:
        risk_level = "extreme"
    elif max_score >= 15:
        risk_level = "high"
    elif max_score >= 8:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "overall_risk_score": max_score,
        "risk_level": risk_level,
        "hazards": hazards,
        "action_plan": actions,
    }
