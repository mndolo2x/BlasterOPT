"""
Pareto Ranker Module.
Ranks economic evaluation results by user-selected economic objectives.
"""

from typing import List
from src.economics.margin_calculator import EconomicResult


def rank_by_objective(results: List[EconomicResult], objective: str = "max_margin") -> List[EconomicResult]:
    """
    Ranks economic evaluation results by selected economic objective.

    Parameters:
    -----------
    results : List[EconomicResult]
        List of economic evaluation results.
    objective : str, default='max_margin'
        Objective criterion: 'max_margin' | 'max_npv' | 'min_cost' | 'max_throughput'

    Returns:
    --------
    List[EconomicResult]
        Ranked list of EconomicResult instances.
    """
    if not results:
        return []

    ranked = list(results)

    if objective == "max_margin":
        ranked.sort(key=lambda r: r.margin_usd_per_t, reverse=True)
    elif objective == "max_npv":
        ranked.sort(key=lambda r: r.npv_usd, reverse=True)
    elif objective == "min_cost":
        ranked.sort(key=lambda r: r.cost.total_usd_per_t, reverse=False)
    elif objective == "max_throughput":
        ranked.sort(key=lambda r: float(r.design_params.get("crusher_throughput_tph", 0.0)), reverse=True)
    else:
        ranked.sort(key=lambda r: r.margin_usd_per_t, reverse=True)

    return ranked
