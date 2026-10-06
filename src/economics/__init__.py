"""
Mine-to-Mill Economics Package for BlasterOPT Botswana.
Exports BenchContext, CostBreakdown, RevenueBreakdown, EconomicResult, and evaluation functions.
"""

from src.economics.bench_context import BenchContext, load_bench_context
from src.economics.cost_model import CostBreakdown, compute_cost_breakdown
from src.economics.revenue_model import RevenueBreakdown, compute_revenue
from src.economics.margin_calculator import EconomicResult, evaluate_design, evaluate_pareto_front
from src.economics.pareto_ranker import rank_by_objective
from src.economics.visualizer import (
    plot_cost_breakdown_stacked,
    plot_margin_vs_powder_factor,
    plot_revenue_vs_cost,
    plot_npv_bars,
    plot_sensitivity_tornado,
)

__all__ = [
    "BenchContext",
    "load_bench_context",
    "CostBreakdown",
    "compute_cost_breakdown",
    "RevenueBreakdown",
    "compute_revenue",
    "EconomicResult",
    "evaluate_design",
    "evaluate_pareto_front",
    "rank_by_objective",
    "plot_cost_breakdown_stacked",
    "plot_margin_vs_powder_factor",
    "plot_revenue_vs_cost",
    "plot_npv_bars",
    "plot_sensitivity_tornado",
]
