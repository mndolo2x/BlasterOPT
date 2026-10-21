"""
Constrained Optimization Architecture for Diamond Preservation.

Evaluates candidate design scenarios against baseline parameters and approved constraints,
calculating parameter deltas and flagging unverified or infeasible candidates.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def evaluate_scenario_deltas(
    baseline_params: Dict[str, Any],
    candidate_params: Dict[str, Any],
    approved_limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compares candidate scenario parameters against baseline parameters.

    Returns:
    --------
    Dict[str, Any]
        (parameter_deltas, feasibility_status, warnings)
    """
    deltas: Dict[str, Any] = {}
    warnings: List[str] = []

    # Calculate parameter differences (Candidate - Baseline)
    for key in ["burden_m", "spacing_m", "stemming_m", "powder_factor_kg_m3", "max_charge_per_delay_kg", "hole_diameter_mm"]:
        if key in candidate_params and key in baseline_params:
            try:
                cand_val = float(candidate_params[key])
                base_val = float(baseline_params[key])
                delta_val = cand_val - base_val
                pct_val = (delta_val / base_val) * 100.0 if abs(base_val) > 1e-6 else 0.0

                deltas[key] = {
                    "baseline": round(base_val, 3),
                    "candidate": round(cand_val, 3),
                    "delta": round(delta_val, 3),
                    "percentage_change": round(pct_val, 2),
                }
            except (ValueError, TypeError):
                pass

    # Feasibility validation against approved limits
    feasibility = "Verified" if approved_limits and approved_limits.get("is_verified", False) else "Unverified"

    if not approved_limits or not approved_limits.get("is_verified", False):
        warnings.append("Feasibility unverified: Site-specific approved diamond safety limits are unverified.")

    # Physical geometry checks
    cand_b = float(candidate_params.get("burden_m", baseline_params.get("burden_m", 6.0)))
    cand_s = float(candidate_params.get("spacing_m", baseline_params.get("spacing_m", 7.0)))
    cand_pf = float(candidate_params.get("powder_factor_kg_m3", baseline_params.get("powder_factor_kg_m3", 0.65)))

    if cand_s < cand_b:
        feasibility = "Rejected"
        warnings.append("Geometry Violation: Spacing must be greater than or equal to burden.")

    if cand_s > 1.5 * cand_b:
        feasibility = "Rejected"
        warnings.append("Geometry Violation: Spacing must not exceed 1.5 × burden.")

    if cand_pf > 1.20:
        warnings.append("High Powder Factor Warning: Powder factor > 1.20 kg/m³ increases shock wave breakage risk.")

    return {
        "deltas": deltas,
        "feasibility_status": feasibility,
        "warnings": warnings,
    }
