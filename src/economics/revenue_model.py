"""
Revenue Model Module.
Computes revenue based on bench tonnage, ore grade, commodity price, and fragmentation-dependent recovery.
"""

import numpy as np
from pydantic import BaseModel, Field
from src.economics.bench_context import BenchContext


class RevenueBreakdown(BaseModel):
    """Pydantic model representing revenue breakdown per tonne and bench total."""

    tonnes: float = Field(..., description="Bench total ore tonnes")
    grade: float = Field(..., description="Ore grade")
    recovery_pct: float = Field(..., description="Plant recovery percentage (0-100%)")
    commodity_price: float = Field(..., description="Commodity selling price USD")
    revenue_usd_per_t: float = Field(..., description="Revenue generated per tonne USD/t")
    total_revenue_usd: float = Field(..., description="Total revenue generated for bench USD")


def compute_revenue(design: dict, bench: BenchContext) -> RevenueBreakdown:
    """
    Computes revenue for a candidate blast design and bench context.

    Parameters:
    -----------
    design : dict
        Candidate blast design dictionary.
    bench : BenchContext
        Bench operational context.

    Returns:
    --------
    RevenueBreakdown
        Validated RevenueBreakdown instance.
    """
    d80_mm = float(design.get("d80_mm", design.get("fragmentation_d80_cm", 25.0) * 10.0 if "fragmentation_d80_cm" in design else 250.0))

    # Recovery curve formula: finer fragmentation (smaller D80) -> higher liberation & recovery
    # Reference D80 = 250mm -> 88% recovery baseline; 150mm -> 94% recovery; 400mm -> 80% recovery
    base_recovery = 88.0
    recovery_pct = float(np.clip(base_recovery + (250.0 - d80_mm) * 0.04, 70.0, 98.0))

    # Revenue $/t = grade * (recovery_pct / 100) * price
    revenue_usd_per_t = float(bench.grade * (recovery_pct / 100.0) * bench.commodity_price)
    total_revenue = float(revenue_usd_per_t * bench.tonnes)

    return RevenueBreakdown(
        tonnes=round(bench.tonnes, 1),
        grade=round(bench.grade, 4),
        recovery_pct=round(recovery_pct, 2),
        commodity_price=round(bench.commodity_price, 2),
        revenue_usd_per_t=round(revenue_usd_per_t, 2),
        total_revenue_usd=round(total_revenue, 2),
    )
