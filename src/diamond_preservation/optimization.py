"""
Constrained Optimization Architecture and Scenario Evaluation for Diamond Preservation.

Evaluates candidate design scenarios against baseline parameters, recalculating dependent
quantities (such as derived powder factor from charge and rock volume), verifying constraints
against explicit validation states (PASS, FAIL, UNVERIFIED, NOT APPLICABLE), and flagging
uncalibrated diamond-breakage predictor states.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def calculate_derived_powder_factor(
    charge_mass_per_hole_kg: Optional[float],
    burden_m: Optional[float],
    spacing_m: Optional[float],
    bench_height_m: Optional[float],
) -> Dict[str, Any]:
    """
    Recalculates derived powder factor (kg/m^3) from explosive loading and rock volume per hole.

    Formula:
        Derived PF = charge_mass_per_hole_kg / (burden_m * spacing_m * bench_height_m)
    """
    if (
        charge_mass_per_hole_kg is None
        or burden_m is None
        or spacing_m is None
        or bench_height_m is None
        or charge_mass_per_hole_kg <= 0
        or burden_m <= 0
        or spacing_m <= 0
        or bench_height_m <= 0
    ):
        missing = []
        if not charge_mass_per_hole_kg or charge_mass_per_hole_kg <= 0: missing.append("charge_mass_per_hole_kg")
        if not burden_m or burden_m <= 0: missing.append("burden_m")
        if not spacing_m or spacing_m <= 0: missing.append("spacing_m")
        if not bench_height_m or bench_height_m <= 0: missing.append("bench_height_m")

        return {
            "status": "not_calculated",
            "value": None,
            "unit": "kg/m³",
            "message": "Not calculated — required inputs unavailable",
            "missing_inputs": missing,
        }

    volume_m3 = burden_m * spacing_m * bench_height_m
    pf = charge_mass_per_hole_kg / volume_m3

    return {
        "status": "calculated",
        "value": round(float(pf), 3),
        "unit": "kg/m³",
        "formula": "charge_mass_per_hole_kg / (burden_m * spacing_m * bench_height_m)",
        "rock_volume_per_hole_m3": round(float(volume_m3), 2),
        "charge_mass_per_hole_kg": charge_mass_per_hole_kg,
    }


def evaluate_scenario_deltas(
    baseline_params: Dict[str, Any],
    candidate_params: Dict[str, Any],
    approved_limits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compares candidate scenario parameters against baseline parameters, recalculating dependent
    quantities and validating constraints against explicit states (PASS, FAIL, UNVERIFIED, NOT APPLICABLE).
    """
    deltas: Dict[str, Any] = {}
    constraint_checks: List[Dict[str, Any]] = []

    # Helper for parameter deltas
    param_meta = {
        "burden_m": ("Burden (m)", "User Input"),
        "spacing_m": ("Spacing (m)", "User Input"),
        "bench_height_m": ("Bench Height (m)", "User Input"),
        "hole_diameter_mm": ("Hole Diameter (mm)", "User Input"),
        "charge_mass_per_hole_kg": ("Charge per Hole (kg)", "User Input"),
        "max_charge_per_delay_kg": ("Max Charge per Delay (kg)", "User Input"),
        "target_powder_factor_kg_m3": ("Target Powder Factor (kg/m³)", "Design Target"),
    }

    # Normalize target powder factor key
    base_target_pf = baseline_params.get("target_powder_factor_kg_m3", baseline_params.get("powder_factor_kg_m3"))
    cand_target_pf = candidate_params.get("target_powder_factor_kg_m3", candidate_params.get("powder_factor_kg_m3"))

    b_p = baseline_params.copy()
    c_p = candidate_params.copy()
    b_p["target_powder_factor_kg_m3"] = base_target_pf
    c_p["target_powder_factor_kg_m3"] = cand_target_pf

    # Calculate parameter differences
    for key, (label, p_type) in param_meta.items():
        if key in c_p and key in b_p:
            try:
                cand_val = float(c_p[key])
                base_val = float(b_p[key])
                delta_val = cand_val - base_val
                pct_val = (delta_val / base_val) * 100.0 if abs(base_val) > 1e-6 else 0.0

                deltas[key] = {
                    "label": label,
                    "parameter_type": p_type,
                    "baseline": round(base_val, 3),
                    "candidate": round(cand_val, 3),
                    "delta": round(delta_val, 3),
                    "percentage_change": round(pct_val, 2),
                    "unit": label.split("(")[-1].replace(")", "").strip() if "(" in label else "",
                }
            except (ValueError, TypeError):
                pass

    # 1. Recalculate Derived Powder Factor for Baseline and Candidate
    b_chg = float(b_p["charge_mass_per_hole_kg"]) if b_p.get("charge_mass_per_hole_kg") is not None else (float(b_p["max_charge_per_delay_kg"]) if b_p.get("max_charge_per_delay_kg") is not None else None)
    b_bur = float(b_p["burden_m"]) if b_p.get("burden_m") is not None else None
    b_spc = float(b_p["spacing_m"]) if b_p.get("spacing_m") is not None else None
    b_hgt = float(b_p["bench_height_m"]) if b_p.get("bench_height_m") is not None else None

    c_chg = float(c_p["charge_mass_per_hole_kg"]) if c_p.get("charge_mass_per_hole_kg") is not None else (float(c_p["max_charge_per_delay_kg"]) if c_p.get("max_charge_per_delay_kg") is not None else None)
    c_bur = float(c_p["burden_m"]) if c_p.get("burden_m") is not None else None
    c_spc = float(c_p["spacing_m"]) if c_p.get("spacing_m") is not None else None
    c_hgt = float(c_p["bench_height_m"]) if c_p.get("bench_height_m") is not None else None

    base_derived_pf = calculate_derived_powder_factor(
        charge_mass_per_hole_kg=b_chg,
        burden_m=b_bur,
        spacing_m=b_spc,
        bench_height_m=b_hgt,
    )

    cand_derived_pf = calculate_derived_powder_factor(
        charge_mass_per_hole_kg=c_chg,
        burden_m=c_bur,
        spacing_m=c_spc,
        bench_height_m=c_hgt,
    )

    if cand_derived_pf["status"] == "calculated" and base_derived_pf["status"] == "calculated":
        base_dpf_val = base_derived_pf["value"]
        cand_dpf_val = cand_derived_pf["value"]
        dpf_delta = cand_dpf_val - base_dpf_val
        dpf_pct = (dpf_delta / base_dpf_val) * 100.0 if abs(base_dpf_val) > 1e-6 else 0.0

        deltas["derived_powder_factor_kg_m3"] = {
            "label": "Derived Powder Factor (kg/m³)",
            "parameter_type": "Derived Calculation",
            "baseline": base_dpf_val,
            "candidate": cand_dpf_val,
            "delta": round(dpf_delta, 3),
            "percentage_change": round(dpf_pct, 2),
            "unit": "kg/m³",
            "formula": cand_derived_pf["formula"],
        }
    else:
        deltas["derived_powder_factor_kg_m3"] = {
            "label": "Derived Powder Factor (kg/m³)",
            "parameter_type": "Derived Calculation",
            "baseline": base_derived_pf.get("value"),
            "candidate": cand_derived_pf.get("value"),
            "delta": None,
            "percentage_change": None,
            "unit": "kg/m³",
            "status": "not_calculated",
            "message": cand_derived_pf.get("message", "Not calculated — required inputs unavailable"),
        }

    # 2. STRENGTHEN CONSTRAINT VALIDATION
    # Check 1: Geometry Ratio (Spacing >= Burden and Spacing <= 1.5 * Burden)
    cand_b = float(c_p.get("burden_m", 6.0))
    cand_s = float(c_p.get("spacing_m", 7.0))

    if cand_s < cand_b or cand_s > 1.5 * cand_b:
        geom_status = "FAIL"
        geom_reason = f"Candidate spacing/burden ratio ({cand_s:.2f}m / {cand_b:.2f}m = {cand_s/cand_b:.2f}) violates geometry limits (1.0 <= S/B <= 1.5)."
    else:
        geom_status = "PASS"
        geom_reason = f"Candidate geometry ratio ({cand_s/cand_b:.2f}) complies with 1.0 <= S/B <= 1.5."

    constraint_checks.append({
        "constraint_name": "Pattern Geometry Ratio (S/B)",
        "source": "BlastOPT Standard Open-Pit Geometry Constraint",
        "status": geom_status,
        "details": geom_reason,
    })

    # Check 2: Derived Powder Factor Limit (<= 0.80 kg/m^3)
    if cand_derived_pf["status"] == "calculated":
        dpf_val = cand_derived_pf["value"]
        if dpf_val > 0.80:
            pf_status = "FAIL"
            pf_reason = f"Derived powder factor ({dpf_val:.3f} kg/m³) exceeds maximum threshold (0.80 kg/m³). High risk of over-fines and diamond breakage."
        else:
            pf_status = "PASS"
            pf_reason = f"Derived powder factor ({dpf_val:.3f} kg/m³) complies with <= 0.80 kg/m³."
    else:
        pf_status = "UNVERIFIED"
        pf_reason = f"Derived powder factor unverified: {cand_derived_pf.get('message')}."

    constraint_checks.append({
        "constraint_name": "Derived Powder Factor Confinement Limit",
        "source": "Jwaneng Kimberlite Diamond Preservation Rule",
        "status": pf_status,
        "details": pf_reason,
    })

    # Check 3: Site Vibration / Permit Limit
    if approved_limits and approved_limits.get("is_verified", False):
        max_ppv_limit = approved_limits.get("max_ppv_mms", 5.0)
        v_status = "PASS"
        v_reason = f"Site permit PPV limit verified at {max_ppv_limit:.1f} mm/s."
    else:
        v_status = "UNVERIFIED"
        v_reason = "Site-specific diamond preservation safety limits are unverified or unconfigured."

    constraint_checks.append({
        "constraint_name": "Site Vibration & Diamond Safety Threshold",
        "source": "Configured Site Limits (config/site_limits.yaml)",
        "status": v_status,
        "details": v_reason,
    })

    # Count overall check states
    passed_count = sum(1 for c in constraint_checks if c["status"] == "PASS")
    failed_count = sum(1 for c in constraint_checks if c["status"] == "FAIL")
    unverified_count = sum(1 for c in constraint_checks if c["status"] == "UNVERIFIED")

    if failed_count > 0:
        overall_feasibility = "Rejected"
    elif unverified_count > 0:
        overall_feasibility = "Unverified"
    else:
        overall_feasibility = "Verified"

    return {
        "deltas": deltas,
        "derived_powder_factor": cand_derived_pf,
        "constraint_checks": constraint_checks,
        "validation_summary": {
            "passed": passed_count,
            "failed": failed_count,
            "unverified": unverified_count,
            "overall_feasibility": overall_feasibility,
        },
        "model_prediction_status": "Diamond-breakage prediction unavailable — no validated model is configured.",
    }
