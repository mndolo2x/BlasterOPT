"""
Environmental Impact Assessment (EIA) index matrix per ISO 14001 / UN EP guidelines.
"""

from typing import Dict, Any


def evaluate_environmental_impact(
    ppv_mms: float,
    airblast_dbl: float,
    flyrock_m: float,
    pm10_ug_m3: float,
    nox_ppm: float,
) -> Dict[str, Any]:
    """
    Evaluate comprehensive Environmental Impact Assessment (EIA) index across vibration, airblast, flyrock, dust, and toxic gas.

    Scoring (0-100 EIA Index):
        EIA Index = 0.25 * S_ppv + 0.25 * S_airblast + 0.20 * S_flyrock + 0.15 * S_dust + 0.15 * S_gas

    Returns:
        {
            "eia_score": float,
            "impact_class": str,  # 'low', 'moderate', 'high', 'severe'
            "sub_scores": dict,
            "compliance_summary": str,
        }
    """
    s_ppv = min(100.0, (ppv_mms / 10.0) * 100.0)
    s_air = min(100.0, (airblast_dbl / 120.0) * 100.0)
    s_fly = min(100.0, (flyrock_m / 250.0) * 100.0)
    s_dust = min(100.0, (pm10_ug_m3 / 150.0) * 100.0)
    s_gas = min(100.0, (nox_ppm / 5.0) * 100.0)

    eia = 0.25 * s_ppv + 0.25 * s_air + 0.20 * s_fly + 0.15 * s_dust + 0.15 * s_gas

    if eia > 85.0:
        cls = "severe"
        summary = "CRITICAL EIA ALARM: Multiple environmental parameters exceed statutory limits."
    elif eia > 65.0:
        cls = "high"
        summary = "HIGH EIA IMPACT: Close to environmental compliance boundaries."
    elif eia > 45.0:
        cls = "moderate"
        summary = "MODERATE EIA IMPACT: Environmental parameters within operational norms."
    else:
        cls = "low"
        summary = "LOW EIA IMPACT: Excellent environmental stewardship."

    return {
        "eia_score": round(eia, 1),
        "impact_class": cls,
        "sub_scores": {
            "vibration_score": round(s_ppv, 1),
            "airblast_score": round(s_air, 1),
            "flyrock_score": round(s_fly, 1),
            "dust_score": round(s_dust, 1),
            "toxic_gas_score": round(s_gas, 1),
        },
        "compliance_summary": summary,
    }
