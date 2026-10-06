"""
Mine-to-Mill Cost Breakdown Module.
Computes drilling, blasting, digging, hauling, crushing, and processing unit costs ($/t).
"""

import numpy as np
from pydantic import BaseModel, Field
from src.economics.bench_context import BenchContext


class CostBreakdown(BaseModel):
    """Pydantic model representing unit cost components in USD per tonne ($/t)."""

    drill_and_blast_usd_per_t: float = Field(..., description="Drilling & Blasting unit cost $/t")
    digging_usd_per_t: float = Field(..., description="Shovel digging unit cost $/t")
    hauling_usd_per_t: float = Field(..., description="Haul truck haulage unit cost $/t")
    crushing_usd_per_t: float = Field(..., description="Primary crusher unit cost $/t")
    processing_usd_per_t: float = Field(..., description="Milling & recovery processing cost $/t")
    total_usd_per_t: float = Field(..., description="Total Mine-to-Mill unit cost $/t")


def compute_cost_breakdown(design: dict, bench: BenchContext) -> CostBreakdown:
    """
    Computes unit cost breakdown for a candidate blast design design dict.

    Parameters:
    -----------
    design : dict
        Blast design dictionary from Pareto front candidate.
    bench : BenchContext
        Bench operational context.

    Returns:
    --------
    CostBreakdown
        Validated CostBreakdown instance.
    """
    # 1. Drill & Blast cost
    d_b_cost = float(design.get("cost_per_tonne_usd", 1.80))

    d80_mm = float(design.get("d80_mm", design.get("fragmentation_d80_cm", 25.0) * 10.0 if "fragmentation_d80_cm" in design else 250.0))
    tph = float(design.get("crusher_throughput_tph", 1500.0))

    # 2. Digging cost: base $0.45/t + penalty if P80/D80 oversize (>300mm)
    oversize_mm = max(0.0, d80_mm - 300.0)
    digging_cost = 0.45 + (d80_mm / 1000.0) * 0.50 + (oversize_mm / 100.0) * 0.15

    # 3. Hauling cost: base $0.80/t + muckpile penalty
    hauling_cost = 0.80 + (d80_mm / 1000.0) * 0.40

    # 4. Crushing cost: function of D80 size and throughput (t/h)
    crushing_cost = 0.35 + (d80_mm / 500.0) * 0.30 + max(0.0, (2000.0 - tph) / 2000.0) * 0.20

    # 5. Processing cost: base $2.50/t
    processing_cost = 2.50 + (d80_mm / 400.0) * 0.50

    total_cost = d_b_cost + digging_cost + hauling_cost + crushing_cost + processing_cost

    return CostBreakdown(
        drill_and_blast_usd_per_t=round(d_b_cost, 2),
        digging_usd_per_t=round(digging_cost, 2),
        hauling_usd_per_t=round(hauling_cost, 2),
        crushing_usd_per_t=round(crushing_cost, 2),
        processing_usd_per_t=round(processing_cost, 2),
        total_usd_per_t=round(total_cost, 2),
    )
