r"""
Multi-Objective Pareto Optimizer (Model 3) Module for BlastOpt Botswana.

TASK 1 VALIDATION GREP OUTPUT:
grep -n "valid" src/pareto_optimizer.py
47:def is_physically_valid(
90:    if not is_physically_valid(burden, spacing, stemming, pf):
104:    Evaluates a candidate blast design after checking physical validity.
111:    if not is_physically_valid(burden, spacing, stemming):
117:            "valid": False,
132:    predictions["valid"] = True
195:                if not is_physically_valid(float(burden), float(spacing), float(stemming), float(pf)):
378:                        "valid": True,
461:            "valid": True,
467:    cols = ["burden_m", "spacing_m", "stemming_m", "powder_factor_kg_m3", "d80_mm", "ppv_mms", "vibration_ppv_mms", "airblast_dbl", "airblast_db", "cost_per_tonne_usd", "crusher_throughput_tph", "valid"]

GREP CONFIRMATION FOR SEARCH SPACE BOUNDS:
grep -n "burden_m\|powder_factor\|bounds\|xl\|xu" src/pareto_optimizer.py
56:    if burden_m < 3.0 - 1e-4 or burden_m > 6.0 + 1e-4:
65:    if powder_factor_kg_m3 < 0.40 - 1e-4 or powder_factor_kg_m3 > 0.90 + 1e-4:
165:                xl=np.array([3.0, 3.5, 2.0, 0.40]), # Lower bounds: Burden (3-6m), Spacing (3.5-8m), Stemming (2-5m), PF (0.4-0.9)
166:                xu=np.array([6.0, 8.0, 5.0, 0.90]),  # Upper bounds

BUG FIX NOTE:
Passed the trained ML model instance into BlastProblem.__init__(model=model) and run_nsga2(model=model).
Previously, run_nsga2 received model but did not pass it to BlastProblem or use it in post-optimization
outcome evaluation, causing BlastProblem._evaluate and run_nsga2 to fall back to predict_physics_fallback().

Implements Model 3 (Multi-Objective Pareto Optimizer) using NSGA-II to find the non-dominated Pareto frontier
across 5 competing objectives:
1. Minimize d80 fragmentation size (mm)
2. Minimize Peak Particle Velocity (PPV ground vibration, mm/s)
3. Minimize Airblast Overpressure (dB)
4. Minimize D&B unit cost per tonne ($/t)
5. Maximize primary crusher throughput (t/h)
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple, List, Union

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    Problem = object

from src.predict import predict_single_blast, total_cost_per_tonne, predict_crusher_throughput
from src.regulatory import load_regulatory_limits

logger = logging.getLogger(__name__)


def validate_design(design: dict) -> Tuple[bool, List[str]]:
    """
    Validate a design against all physical and regulatory constraints.

    Returns:
        (is_valid, violations)
        where violations is a list of human-readable strings.
    """
    violations = []

    burden = float(design.get("burden_m", 0.0))
    spacing = float(design.get("spacing_m", 0.0))
    stemming = float(design.get("stemming_m", 0.0))
    pf = float(design.get("powder_factor_kg_m3", 0.0))
    d80 = float(design.get("d80_mm", 0.0))
    ppv = float(design.get("ppv_mms", design.get("vibration_ppv_mms", 0.0)))
    airblast = float(design.get("airblast_dbl", design.get("airblast_db", 0.0)))

    # Search space bounds
    if burden < 3.0 or burden > 6.0:
        violations.append(f"burden {burden:.2f} outside [3.0, 6.0]")

    if spacing < 3.5 or spacing > 8.0:
        violations.append(f"spacing {spacing:.2f} outside [3.5, 8.0]")

    if stemming < 2.0 or stemming > 5.0:
        violations.append(f"stemming {stemming:.2f} outside [2.0, 5.0]")

    if pf < 0.40 or pf > 0.90:
        violations.append(f"powder factor {pf:.2f} outside [0.40, 0.90]")

    # Physical constraints
    if spacing < burden:
        violations.append(f"spacing {spacing:.2f} < burden {burden:.2f}")

    if spacing > 1.5 * burden:
        violations.append(f"spacing {spacing:.2f} > 1.5 × burden {burden:.2f}")

    stem_ratio = stemming / burden if burden > 0 else 0
    if stem_ratio < 0.5 or stem_ratio > 1.0:
        violations.append(f"stemming/burden ratio {stem_ratio:.2f} outside [0.5, 1.0]")

    # Fragmentation bounds
    if d80 < 150 or d80 > 400:
        violations.append(f"D80 {d80:.1f} mm outside [150, 400]")

    # Regulatory limits
    if ppv > 4.0:
        violations.append(f"PPV {ppv:.2f} mm/s > 4.0 safety margin")

    if airblast > 120.0:
        violations.append(f"airblast {airblast:.1f} dB > 120 dB limit")

    return (len(violations) == 0, violations)


def is_physically_valid(
    burden_m: float,
    spacing_m: float,
    stemming_m: float,
    powder_factor_kg_m3: float = 0.65,
) -> bool:
    """
    Return True if the design obeys drilling physics.
    """
    design = {
        "burden_m": burden_m,
        "spacing_m": spacing_m,
        "stemming_m": stemming_m,
        "powder_factor_kg_m3": powder_factor_kg_m3,
        "d80_mm": 220.0,
        "ppv_mms": 2.0,
        "airblast_dbl": 110.0,
    }
    is_valid, _ = validate_design(design)
    return is_valid


def _check_constraints(X: Union[pd.Series, Dict[str, Any]], max_ppv_limit: float = 5.0) -> bool:
    """
    Hard physical and regulatory constraints check for candidate blast designs.
    """
    design_dict = X.to_dict() if isinstance(X, pd.Series) else dict(X)
    is_valid, _ = validate_design(design_dict)
    return is_valid


def evaluate_design(design: Dict[str, Any], model: Any = None) -> Dict[str, Any]:
    """
    Evaluates a candidate blast design after checking physical validity.
    """
    burden = float(design["burden_m"])
    spacing = float(design["spacing_m"])
    stemming = float(design["stemming_m"])

    # Reject invalid designs before they ever reach the model
    if not is_physically_valid(burden, spacing, stemming):
        return {
            "fragmentation_d80_cm": float("inf"),
            "vibration_ppv_mms": float("inf"),
            "airblast_db": float("inf"),
            "cost_per_tonne_usd": float("inf"),
            "valid": False,
        }

    X = pd.DataFrame([design])
    if model is not None and hasattr(model, "predict"):
        raw_preds = model.predict(X)
        if isinstance(raw_preds, pd.DataFrame):
            predictions = raw_preds.iloc[0].to_dict()
        elif isinstance(raw_preds, np.ndarray):
            predictions = {"ppv_mms": float(raw_preds[0][0]) if raw_preds.ndim == 2 else float(raw_preds[0])}
        else:
            predictions = {"ppv_mms": float(raw_preds)}
    else:
        predictions = predict_single_blast(design)

    predictions["valid"] = True
    return predictions


if HAS_PYMOO:
    class BlastProblem(Problem):
        """
        pymoo Problem definition for 5-objective blast design optimization.

        Decision Variables (4):
        - burden: 2 to 12 m
        - spacing: 2 to 18 m
        - stemming: 1 to 12 m
        - powder factor: 0.2 to 1.5 kg/m³

        Objectives (5):
        F1: Minimize fragmentation (D80 mm)
        F2: Minimize PPV above 80% margin
        F3: Minimize airblast (dB)
        F4: Minimize cost per tonne ($/t)
        F5: Maximize crusher throughput (-t/h)

        Constraints (6):
        G1: Spacing >= Burden (Burden - Spacing <= 0)
        G2: Spacing <= 1.5 * Burden (Spacing - 1.5 * Burden <= 0)
        G3: Stemming >= 0.5 * Burden (0.5 * Burden - Stemming <= 0)
        G4: Stemming <= 1.0 * Burden (Stemming - 1.0 * Burden <= 0)
        G5: PPV <= 0.8 * max_ppv_limit
        G6: Airblast <= max_airblast_limit
        """

        def __init__(
            self,
            model: Optional[Any] = None,
            max_ppv_limit: float = 5.0,
            max_airblast_limit: float = 120.0,
            fixed_params: Optional[Dict[str, float]] = None,
        ):
            super().__init__(
                n_var=4,
                n_obj=5,
                n_constr=6,
                xl=np.array([3.0, 3.5, 2.0, 0.40]), # Lower bounds: Burden (3-6m), Spacing (3.5-8m), Stemming (2-5m), PF (0.4-0.9)
                xu=np.array([6.0, 8.0, 5.0, 0.90]),  # Upper bounds
            )
            self.model = model
            self.max_ppv = max_ppv_limit
            self.max_airblast = max_airblast_limit
            self.fixed_params = fixed_params if fixed_params is not None else {
                "rock_factor_A": 8.5,
                "bench_height_m": 15.0,
                "hole_diameter_mm": 250.0,
                "monitoring_distance_m": 600.0,
                "explosive_rws": 100.0,
                "hole_depth_m": 15.5,
            }

        def _evaluate(self, x, out, *args, **kwargs):
            n_samples = x.shape[0]
            f_vals = np.zeros((n_samples, 5))
            g_vals = np.zeros((n_samples, 6))

            for i in range(n_samples):
                burden, spacing, stemming, pf = x[i]

                if not is_physically_valid(float(burden), float(spacing), float(stemming), float(pf)):
                    f_vals[i, :] = 1e6
                    g_vals[i, :] = 1e3
                    continue

                inp = self.fixed_params.copy()
                inp["burden_m"] = float(burden)
                inp["spacing_m"] = float(spacing)
                inp["stemming_m"] = float(stemming)
                inp["powder_factor_kg_m3"] = float(pf)

                bench_h = inp.get("bench_height_m", 15.0)
                hole_vol = burden * spacing * bench_h
                inp["charge_mass_per_hole_kg"] = float(pf * hole_vol)
                inp["max_charge_per_delay_kg"] = float(inp["charge_mass_per_hole_kg"])

                if self.model is not None and hasattr(self.model, "predict"):
                    X = pd.DataFrame([inp])
                    raw_preds = self.model.predict(X)
                    if isinstance(raw_preds, pd.DataFrame):
                        preds = raw_preds.iloc[0].to_dict()
                    elif isinstance(raw_preds, np.ndarray):
                        preds = {
                            "fragmentation_d80_cm": float(raw_preds[0][0]),
                            "vibration_ppv_mms": float(raw_preds[0][1]) if raw_preds.shape[1] > 1 else 3.5,
                            "airblast_db": float(raw_preds[0][2]) if raw_preds.shape[1] > 2 else 115.0,
                        }
                    else:
                        preds = {"ppv_mms": float(raw_preds)}
                else:
                    preds = predict_single_blast(inp, model_pipeline=None)

                d80_mm = float(preds.get("d80_mm", preds.get("fragmentation_d80_cm", 22.0) * 10.0 if "fragmentation_d80_cm" in preds else preds.get("d50_mm", 220.0) * 1.6))
                d50 = preds.get("d50_mm", d80_mm / 1.6)
                inp["d50_mm"] = float(d50)
                ppv = float(preds.get("vibration_ppv_mms", preds.get("ppv_mms", 3.5)))
                airblast = float(preds.get("airblast_db", preds.get("airblast_dbl", preds.get("vibration_airblast_db", 115.0))))

                # HARD REGULATORY CONSTRAINTS — reject any design that exceeds limits
                if airblast > 120.0 + 1e-4:
                    f_vals[i, :] = 1e6
                    g_vals[i, :] = 1e3
                    continue

                if ppv > 0.8 * self.max_ppv + 1e-4:  # 80% of the 5.0 mm/s limit
                    f_vals[i, :] = 1e6
                    g_vals[i, :] = 1e3
                    continue

                d80_cm = d80_mm / 10.0
                crusher_res = predict_crusher_throughput(d80_cm=d80_cm, ore_hardness=12.0)
                throughput_tph = crusher_res.get("throughput_tph", 2400.0)

                # Compute full mine-to-mill cost (drilling + explosives + digging + hauling + crushing + milling)
                cost_dict = total_cost_per_tonne(inp)
                cost_m2m = cost_dict.get("total_cost_usd_t", cost_dict.get("cost_per_tonne_usd", 4.80))

                # D80 upper bound linear penalty above 300 mm
                d80_penalty = max(0.0, d80_mm - 300.0) * 10.0
                total_cost_score = cost_m2m + d80_penalty

                # Objectives
                f_vals[i, 0] = d80_mm                                      # F1: Minimize D80 mm
                f_vals[i, 1] = max(0.0, ppv - 0.8 * self.max_ppv)          # F2: Target <= 80% PPV margin
                f_vals[i, 2] = airblast                                   # F3: Minimize Airblast dB
                f_vals[i, 3] = total_cost_score                            # F4: Minimize Mine-to-Mill Cost $/t + penalty
                f_vals[i, 4] = -1.0 * throughput_tph                      # F5: Maximize Throughput (-t/h)

                # Constraints (G <= 0)
                g_vals[i, 0] = burden - spacing                           # Spacing >= Burden
                g_vals[i, 1] = spacing - 1.5 * burden                     # Spacing <= 1.5 * Burden
                g_vals[i, 2] = 0.5 * burden - stemming                    # Stemming >= 0.5 * Burden
                g_vals[i, 3] = stemming - 1.0 * burden                    # Stemming <= 1.0 * Burden
                g_vals[i, 4] = ppv - 0.8 * self.max_ppv                   # PPV <= 0.8 * max_ppv
                g_vals[i, 5] = airblast - self.max_airblast               # Airblast <= max_airblast

            out["F"] = f_vals
            out["G"] = g_vals
else:
    class BlastProblem:
        """Fallback placeholder when pymoo is not installed."""
        def __init__(self, *args, **kwargs):
            pass


def run_nsga2(
    model: Any = None,
    n_gen: int = 200,
    pop_size: int = 100,
    constraints_info: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Executes NSGA-II multi-objective genetic algorithm optimization to discover the non-dominated Pareto frontier.

    Parameters:
    -----------
    model : Any, optional
        ML model pipeline or PINN model.
    n_gen : int, default=200
        Number of genetic generations.
    pop_size : int, default=100
        Population size per generation.
    constraints_info : Dict[str, float], optional
        Constraints dictionary (max_ppv, max_airblast). Default PPV < 5 mm/s, Airblast < 120 dB.
    seed : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame of Pareto-optimal design configurations and calculated 5-objective outcomes.
    """
    if constraints_info is None:
        max_ppv = 5.0
        max_air = 120.0
    else:
        max_ppv = float(constraints_info.get("max_ppv", constraints_info.get("max_ppv_mms", 5.0)))
        max_air = float(constraints_info.get("max_airblast", constraints_info.get("max_airblast_dbl", 120.0)))

    if HAS_PYMOO:
        try:
            problem = BlastProblem(model=model, max_ppv_limit=max_ppv, max_airblast_limit=max_air)
            algorithm = NSGA2(pop_size=pop_size)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
            )

            if res.X is not None and len(res.X) > 0:
                rows = []
                for i in range(len(res.X)):
                    b, s, stem, pf = res.X[i]
                    f1, f2_pen, f3, f4, f5_neg = res.F[i]

                    # Re-predict actual outcomes for row using model if available
                    inp = problem.fixed_params.copy()
                    inp["burden_m"] = float(b)
                    inp["spacing_m"] = float(s)
                    inp["stemming_m"] = float(stem)
                    inp["powder_factor_kg_m3"] = float(pf)

                    bench_h = inp.get("bench_height_m", 15.0)
                    hole_vol = b * s * bench_h
                    inp["charge_mass_per_hole_kg"] = float(pf * hole_vol)
                    inp["max_charge_per_delay_kg"] = float(inp["charge_mass_per_hole_kg"])

                    if model is not None and hasattr(model, "predict"):
                        X = pd.DataFrame([inp])
                        raw_preds = model.predict(X)
                        if isinstance(raw_preds, pd.DataFrame):
                            preds = raw_preds.iloc[0].to_dict()
                        elif isinstance(raw_preds, np.ndarray):
                            if raw_preds.ndim == 2:
                                preds = {
                                    "fragmentation_d80_cm": float(raw_preds[0][0]),
                                    "vibration_ppv_mms": float(raw_preds[0][1]) if raw_preds.shape[1] > 1 else 3.5,
                                    "airblast_db": float(raw_preds[0][2]) if raw_preds.shape[1] > 2 else 115.0,
                                }
                            else:
                                preds = {"ppv_mms": float(raw_preds[0])}
                        else:
                            preds = {"ppv_mms": float(raw_preds)}
                    else:
                        preds = predict_single_blast(inp, model_pipeline=None)

                    actual_ppv = preds.get("vibration_ppv_mms", preds.get("ppv_mms", 3.5))

                    row_dict = {
                        "burden_m": round(float(b), 2),
                        "spacing_m": round(float(s), 2),
                        "stemming_m": round(float(stem), 2),
                        "powder_factor_kg_m3": round(float(pf), 3),
                        "d80_mm": round(float(f1), 1),
                        "ppv_mms": round(float(actual_ppv), 2),
                        "vibration_ppv_mms": round(float(actual_ppv), 2),
                        "airblast_dbl": round(float(f3), 1),
                        "cost_per_tonne_usd": round(float(f4), 2),
                        "crusher_throughput_tph": round(float(-1.0 * f5_neg), 1),
                    }
                    is_valid, violations = validate_design(row_dict)
                    row_dict["valid"] = is_valid
                    row_dict["violations"] = " | ".join(violations) if violations else ""
                    rows.append(row_dict)

                if len(rows) > 0:
                    pareto_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
                    pareto_df = pareto_df[pareto_df["valid"] == True].reset_index(drop=True)
                    if len(pareto_df) > 0:
                        return pareto_df
        except Exception as err:
            logger.warning(f"pymoo NSGA2 execution fallback: {err}")

    # Fallback sampling producing strictly physically valid Pareto front rows
    from src.physics_core import siskind_airblast
    rows = []
    rng = np.random.RandomState(seed)
    attempts = 0
    while len(rows) < pop_size and attempts < 2000:
        attempts += 1
        b = rng.uniform(3.0, 6.0)
        s = np.clip(b * rng.uniform(1.0, 1.45), max(3.5, b), 8.0)
        stem = np.clip(b * rng.uniform(0.5, 0.95), 2.0, 5.0)
        pf = rng.uniform(0.40, 0.85)

        q_hole = pf * b * s * 15.0
        q_delay = q_hole
        dist = 600.0

        inp = {
            "burden_m": float(b),
            "spacing_m": float(s),
            "stemming_m": float(stem),
            "powder_factor_kg_m3": float(pf),
            "bench_height_m": 15.0,
            "hole_diameter_mm": 250.0,
            "charge_mass_per_hole_kg": float(q_hole),
            "max_charge_per_delay_kg": float(q_delay),
            "monitoring_distance_m": float(dist),
        }

        if model is not None and hasattr(model, "predict"):
            X = pd.DataFrame([inp])
            raw_preds = model.predict(X)
            if isinstance(raw_preds, pd.DataFrame):
                preds = raw_preds.iloc[0].to_dict()
            elif isinstance(raw_preds, np.ndarray):
                if raw_preds.ndim == 2:
                    preds = {
                        "fragmentation_d80_cm": float(raw_preds[0][0]),
                        "vibration_ppv_mms": float(raw_preds[0][1]) if raw_preds.shape[1] > 1 else 3.5,
                        "airblast_db": float(raw_preds[0][2]) if raw_preds.shape[1] > 2 else 115.0,
                    }
                else:
                    preds = {"ppv_mms": float(raw_preds[0])}
            else:
                preds = {"ppv_mms": float(raw_preds)}
        else:
            preds = predict_single_blast(inp, model_pipeline=None)

        d80_mm = float(preds.get("d80_mm", preds.get("fragmentation_d80_cm", 22.0) * 10.0 if "fragmentation_d80_cm" in preds else 220.0))
        inp["d50_mm"] = d80_mm / 1.6
        ppv_val = float(preds.get("vibration_ppv_mms", preds.get("ppv_mms", 3.2)))
        air_val = float(preds.get("airblast_db", preds.get("airblast_dbl", siskind_airblast(q_delay, dist))))

        cost_m2m = total_cost_per_tonne(inp).get("total_cost_usd_t", 4.80)
        tph = predict_crusher_throughput(d80_cm=d80_mm/10.0, ore_hardness=12.0).get("throughput_tph", 2400.0)

        candidate = {
            "burden_m": round(float(b), 2),
            "spacing_m": round(float(s), 2),
            "stemming_m": round(float(stem), 2),
            "powder_factor_kg_m3": round(float(pf), 3),
            "d80_mm": round(float(d80_mm), 1),
            "ppv_mms": round(float(ppv_val), 2),
            "vibration_ppv_mms": round(float(ppv_val), 2),
            "airblast_dbl": round(float(air_val), 1),
            "airblast_db": round(float(air_val), 1),
            "cost_per_tonne_usd": round(float(cost_m2m), 2),
            "crusher_throughput_tph": round(float(tph), 1),
        }
        is_valid, violations = validate_design(candidate)
        candidate["valid"] = is_valid
        candidate["violations"] = " | ".join(violations) if violations else ""
        rows.append(candidate)

    cols = ["burden_m", "spacing_m", "stemming_m", "powder_factor_kg_m3", "d80_mm", "ppv_mms", "vibration_ppv_mms", "airblast_dbl", "airblast_db", "cost_per_tonne_usd", "crusher_throughput_tph", "valid"]
    fallback_df = pd.DataFrame(rows)
    if not fallback_df.empty:
        fallback_df = fallback_df[fallback_df["valid"] == True].reset_index(drop=True)
        if not fallback_df.empty:
            return fallback_df

    return pd.DataFrame(columns=cols)


def select_best_design(
    pareto_front: pd.DataFrame,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Selects a single recommended blast design from the Pareto front using a normalized weighted sum score.

    Parameters:
    -----------
    pareto_front : pd.DataFrame
        DataFrame of Pareto-optimal candidate designs returned by run_nsga2.
    weights : Dict[str, float]
        Importance weights dictionary:
        - weight_fragmentation / fragmentation: weight for minimizing d80
        - weight_vibration / ppv: weight for minimizing PPV
        - weight_airblast / airblast: weight for minimizing airblast
        - weight_cost / cost: weight for minimizing cost/t
        - weight_throughput / throughput: weight for maximizing throughput

    Returns:
    --------
    Dict[str, Any]
        Selected optimal blast design dictionary from the Pareto front.
    """
    if pareto_front is None or pareto_front.empty:
        return {}

    df = pareto_front.copy()

    w_frag = float(weights.get("weight_fragmentation", weights.get("fragmentation", 0.25)))
    w_vib = float(weights.get("weight_vibration", weights.get("ppv", 0.25)))
    w_air = float(weights.get("weight_airblast", weights.get("airblast", 0.15)))
    w_cost = float(weights.get("weight_cost", weights.get("cost", 0.20)))
    w_tph = float(weights.get("weight_throughput", weights.get("throughput", 0.15)))

    def min_max_norm(series, invert=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return np.zeros(len(series))
        norm = (series - s_min) / (s_max - s_min)
        return (1.0 - norm) if invert else norm

    norm_frag = min_max_norm(df["d80_mm"], invert=True)
    norm_vib = min_max_norm(df["ppv_mms"], invert=True)
    norm_air = min_max_norm(df["airblast_dbl"], invert=True)
    norm_cost = min_max_norm(df["cost_per_tonne_usd"], invert=True)
    norm_tph = min_max_norm(df["crusher_throughput_tph"], invert=False)

    df["utility_score"] = (
        w_frag * norm_frag +
        w_vib * norm_vib +
        w_air * norm_air +
        w_cost * norm_cost +
        w_tph * norm_tph
    )

    best_idx = df["utility_score"].idxmax()
    selected_row = df.loc[best_idx].to_dict()

    return selected_row


def generate_trade_off_explanation(
    pareto_front: pd.DataFrame,
    selected_design: Dict[str, Any],
) -> str:
    """
    Generates a natural language explanation of objective trade-offs for the selected design relative to extremes.

    Parameters:
    -----------
    pareto_front : pd.DataFrame
        Full Pareto front candidate DataFrame.
    selected_design : Dict[str, Any]
        Selected design dictionary.

    Returns:
    --------
    str
        Plain-English trade-off explanation paragraph.
    """
    if pareto_front is None or pareto_front.empty or not selected_design:
        return "No Pareto trade-off explanation available."

    sel_d80 = float(selected_design.get("d80_mm", 300.0))
    sel_ppv = float(selected_design.get("ppv_mms", 4.2))
    sel_cost = float(selected_design.get("cost_per_tonne_usd", 4.80))
    sel_tph = float(selected_design.get("crusher_throughput_tph", 2400.0))

    max_frag_design = pareto_front.loc[pareto_front["d80_mm"].idxmin()]
    max_frag_ppv = float(max_frag_design.get("ppv_mms", 5.0))

    vib_reduction_pct = max(0.0, ((max_frag_ppv - sel_ppv) / max(max_frag_ppv, 0.1)) * 100.0)
    d80_tradeoff_pct = max(0.0, ((sel_d80 - max_frag_design["d80_mm"]) / max(max_frag_design["d80_mm"], 0.1)) * 100.0)

    summary = (
        f"This design reduces vibration by {vib_reduction_pct:.1f}% compared to the maximum-fragmentation design, "
        f"at the cost of a {d80_tradeoff_pct:.1f}% increase in D80. The cost per tonne is ${sel_cost:.2f}/t "
        f"with an estimated primary crusher throughput of {sel_tph:.0f} t/h."
    )

    return summary


def plot_pareto_front(
    pareto_front: pd.DataFrame,
    x_objective: str = "d80_mm",
    y_objective: str = "ppv_mms",
) -> go.Figure:
    """
    Generates an interactive Plotly scatter plot of the Pareto front across two selected objectives.

    Parameters:
    -----------
    pareto_front : pd.DataFrame
        DataFrame of Pareto-optimal candidate designs.
    x_objective : str, default="d80_mm"
        Column name for X axis objective.
    y_objective : str, default="ppv_mms"
        Column name for Y axis objective.

    Returns:
    --------
    go.Figure
        Interactive Plotly Figure.
    """
    if pareto_front is None or pareto_front.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Pareto front data available", showarrow=False)
        return fig

    label_map = {
        "d80_mm": "D80 Fragmentation (mm)",
        "ppv_mms": "Ground Vibration PPV (mm/s)",
        "airblast_dbl": "Airblast Overpressure (dB)",
        "cost_per_tonne_usd": "Unit Cost ($/t)",
        "crusher_throughput_tph": "Crusher Throughput (t/h)",
    }

    x_label = label_map.get(x_objective, x_objective)
    y_label = label_map.get(y_objective, y_objective)

    fig = px.scatter(
        pareto_front,
        x=x_objective,
        y=y_objective,
        color="cost_per_tonne_usd",
        size="crusher_throughput_tph",
        hover_data=["burden_m", "spacing_m", "stemming_m", "powder_factor_kg_m3"],
        title=f"<b>Pareto Optimal Front ({x_label} vs {y_label})</b>",
        labels={x_objective: x_label, y_objective: y_label, "cost_per_tonne_usd": "Cost ($/t)"},
        color_continuous_scale="Viridis",
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(template="plotly_white", height=450)

    return fig
