"""
Margin Calculator Module.
Evaluates cost, revenue, net margin per tonne ($/t), bench total margin ($), and discounted NPV ($).
"""

import pandas as pd
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.economics.bench_context import BenchContext, load_bench_context
from src.economics.cost_model import CostBreakdown, compute_cost_breakdown
from src.economics.revenue_model import RevenueBreakdown, compute_revenue


class EconomicResult(BaseModel):
    """Pydantic model representing complete economic evaluation results for a design."""

    design_id: int = Field(..., description="Index ID of design candidate")
    design_params: Dict[str, Any] = Field(..., description="Design input geometry and predicted outcomes")
    cost: CostBreakdown = Field(..., description="Cost breakdown")
    revenue: RevenueBreakdown = Field(..., description="Revenue breakdown")
    margin_usd_per_t: float = Field(..., description="Net economic margin per tonne $/t")
    total_margin_usd: float = Field(..., description="Total net margin for bench USD")
    npv_usd: float = Field(..., description="Net Present Value discounted over bench life USD")
    status: str = Field("SAFE", description="Safety and economic status ('SAFE' or 'REQUIRES_REVIEW')")
    review_reasons: List[str] = Field(default_factory=list, description="List of review triggers if status is REQUIRES_REVIEW")


def evaluate_design(design: dict, bench: BenchContext, design_id: int = 0) -> EconomicResult:
    """
    Evaluates economic performance (costs, revenue, margin, NPV) for a single candidate blast design.

    Parameters:
    -----------
    design : dict
        Candidate blast design parameter dictionary.
    bench : BenchContext
        Bench context model instance.
    design_id : int, default=0
        Unique design index or ID.

    Returns:
    --------
    EconomicResult
        Complete economic evaluation result.
    """
    cost_bd = compute_cost_breakdown(design, bench)
    revenue_bd = compute_revenue(design, bench)

    margin_per_t = revenue_bd.revenue_usd_per_t - cost_bd.total_usd_per_t
    total_margin = margin_per_t * bench.tonnes

    # NPV = total_margin / (1 + discount_rate)^bench_life_years
    discount_factor = (1.0 + bench.discount_rate) ** bench.bench_life_years
    npv_usd = total_margin / discount_factor

    status = "SAFE"
    review_reasons = []

    # Flag designs where cost prediction is outside reasonable range ($1.00 - $10.00 / t)
    d_b_cost = cost_bd.drill_and_blast_usd_per_t
    if d_b_cost < 1.00 or d_b_cost > 10.00:
        status = "REQUIRES_REVIEW"
        review_reasons.append(f"D&B cost (${d_b_cost:.2f}/t) outside expected reasonable range [$1.00, $10.00]/t")

    if not design.get("valid", True):
        status = "REQUIRES_REVIEW"
        review_reasons.append(f"Physical or regulatory violations: {design.get('violations', 'Invalid design')}")

    return EconomicResult(
        design_id=design_id,
        design_params=design,
        cost=cost_bd,
        revenue=revenue_bd,
        margin_usd_per_t=round(margin_per_t, 2),
        total_margin_usd=round(total_margin, 2),
        npv_usd=round(npv_usd, 2),
        status=status,
        review_reasons=review_reasons,
    )


def evaluate_pareto_front(pareto_df: pd.DataFrame, bench: BenchContext) -> List[EconomicResult]:
    """
    Evaluates economic performance across all candidate designs in a Pareto front DataFrame.

    Parameters:
    -----------
    pareto_df : pd.DataFrame
        DataFrame of candidate designs from Pareto front.
    bench : BenchContext
        Bench context model instance.

    Returns:
    --------
    List[EconomicResult]
        List of EconomicResult instances for all candidates.
    """
    if pareto_df is None or pareto_df.empty:
        return []

    results = []
    for idx, row in pareto_df.iterrows():
        design_dict = row.to_dict()
        res = evaluate_design(design_dict, bench, design_id=int(idx))
        results.append(res)

    return results
