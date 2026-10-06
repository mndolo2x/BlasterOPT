"""
Unit test suite for Mine-to-Mill Economics Module (src/economics/).
"""

import os
import pandas as pd
import pytest
from src.economics import (
    load_bench_context,
    compute_cost_breakdown,
    compute_revenue,
    evaluate_design,
    evaluate_pareto_front,
    rank_by_objective,
    BenchContext,
    CostBreakdown,
    RevenueBreakdown,
    EconomicResult,
)


def test_bench_context_loads_from_yaml():
    """Test load_bench_context loads valid YAML configs."""
    bench = load_bench_context("jwaneng_bench_14")
    assert bench.bench_id == "jwaneng_bench_14"
    assert bench.site_id == "Jwaneng Mine"
    assert bench.tonnes == 250000.0
    assert bench.grade == 0.42


def test_cost_breakdown_returns_all_components():
    """Test compute_cost_breakdown returns all 5 Mine-to-Mill unit cost components."""
    bench = load_bench_context("jwaneng_bench_14")
    design = {
        "cost_per_tonne_usd": 2.10,
        "d80_mm": 240.0,
        "crusher_throughput_tph": 1500.0,
        "valid": True,
    }

    cost = compute_cost_breakdown(design, bench)
    assert isinstance(cost, CostBreakdown)
    assert cost.drill_and_blast_usd_per_t == 2.10
    assert cost.digging_usd_per_t > 0
    assert cost.hauling_usd_per_t > 0
    assert cost.crushing_usd_per_t > 0
    assert cost.processing_usd_per_t > 0
    assert cost.total_usd_per_t > cost.drill_and_blast_usd_per_t


def test_revenue_increases_with_finer_fragmentation():
    """Test that finer fragmentation (smaller D80) yields higher plant recovery and higher revenue."""
    bench = load_bench_context("jwaneng_bench_14")

    coarse_design = {"d80_mm": 380.0}
    fine_design = {"d80_mm": 180.0}

    rev_coarse = compute_revenue(coarse_design, bench)
    rev_fine = compute_revenue(fine_design, bench)

    assert rev_fine.recovery_pct > rev_coarse.recovery_pct
    assert rev_fine.revenue_usd_per_t > rev_coarse.revenue_usd_per_t


def test_margin_calculator_ranks_correctly():
    """Test margin calculator computes positive net margin and NPV."""
    bench = load_bench_context("jwaneng_bench_14")
    design = {
        "cost_per_tonne_usd": 2.50,
        "d80_mm": 220.0,
        "crusher_throughput_tph": 1600.0,
        "powder_factor_kg_m3": 0.68,
        "valid": True,
    }

    res = evaluate_design(design, bench, design_id=1)
    assert isinstance(res, EconomicResult)
    assert res.margin_usd_per_t > 0
    assert res.npv_usd > 0
    assert res.status == "SAFE"


def test_pareto_ranker_by_max_margin():
    """Test rank_by_objective sorts candidate designs by maximum margin."""
    bench = load_bench_context("jwaneng_bench_14")
    df = pd.DataFrame([
        {"d80_mm": 350.0, "cost_per_tonne_usd": 3.00, "crusher_throughput_tph": 1200.0, "powder_factor_kg_m3": 0.50, "valid": True},
        {"d80_mm": 180.0, "cost_per_tonne_usd": 2.10, "crusher_throughput_tph": 1600.0, "powder_factor_kg_m3": 0.72, "valid": True},
    ])

    evals = evaluate_pareto_front(df, bench)
    ranked = rank_by_objective(evals, objective="max_margin")

    assert len(ranked) == 2
    assert ranked[0].margin_usd_per_t >= ranked[1].margin_usd_per_t
    assert ranked[0].design_id == 1  # Fine design has higher margin


def test_pareto_ranker_by_max_npv():
    """Test rank_by_objective sorts candidate designs by maximum NPV."""
    bench = load_bench_context("jwaneng_bench_14")
    df = pd.DataFrame([
        {"d80_mm": 350.0, "cost_per_tonne_usd": 3.00, "crusher_throughput_tph": 1200.0, "powder_factor_kg_m3": 0.50, "valid": True},
        {"d80_mm": 200.0, "cost_per_tonne_usd": 2.00, "crusher_throughput_tph": 1500.0, "powder_factor_kg_m3": 0.70, "valid": True},
    ])

    evals = evaluate_pareto_front(df, bench)
    ranked = rank_by_objective(evals, objective="max_npv")

    assert ranked[0].npv_usd >= ranked[1].npv_usd


def test_dashboard_handles_empty_pareto_front():
    """Test evaluate_pareto_front returns empty list for empty DataFrame."""
    bench = load_bench_context("jwaneng_bench_14")
    empty_df = pd.DataFrame()
    evals = evaluate_pareto_front(empty_df, bench)
    assert evals == []


def test_out_of_range_cost_flagged_requires_review():
    """Test designs with unreasonable D&B cost are flagged as REQUIRES_REVIEW."""
    bench = load_bench_context("jwaneng_bench_14")
    invalid_cost_design = {
        "cost_per_tonne_usd": 15.00,  # Unreasonable D&B cost (> $10/t)
        "d80_mm": 250.0,
        "valid": True,
    }

    res = evaluate_design(invalid_cost_design, bench, design_id=99)
    assert res.status == "REQUIRES_REVIEW"
    assert any("outside expected reasonable range" in reason for reason in res.review_reasons)
