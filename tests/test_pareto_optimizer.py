"""
Unit tests for Multi-Objective Pareto Optimizer (Model 3) module.
"""

import pytest
import pandas as pd
from src.pareto_optimizer import (
    run_nsga2,
    select_best_design,
    generate_trade_off_explanation,
)


def test_run_nsga2_returns_dataframe_with_correct_columns():
    """Test run_nsga2 returns a non-empty DataFrame with all required objective columns."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)

    assert isinstance(df_pareto, pd.DataFrame)
    assert len(df_pareto) > 0

    expected_cols = [
        "burden_m",
        "spacing_m",
        "stemming_m",
        "powder_factor_kg_m3",
        "d80_mm",
        "ppv_mms",
        "airblast_dbl",
        "cost_per_tonne_usd",
        "crusher_throughput_tph",
    ]

    for col in expected_cols:
        assert col in df_pareto.columns, f"Expected column '{col}' in Pareto front DataFrame"


def test_select_best_design_returns_row_from_pareto_front():
    """Test select_best_design selects a valid row dictionary from the Pareto front."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)

    weights = {
        "weight_fragmentation": 0.30,
        "weight_vibration": 0.30,
        "weight_airblast": 0.10,
        "weight_cost": 0.15,
        "weight_throughput": 0.15,
    }

    selected = select_best_design(df_pareto, weights)

    assert isinstance(selected, dict)
    assert "burden_m" in selected
    assert "d80_mm" in selected
    assert "ppv_mms" in selected
    assert "utility_score" in selected


def test_generate_trade_off_explanation_returns_non_empty_string():
    """Test generate_trade_off_explanation returns non-empty plain English trade-off text."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)
    weights = {"weight_fragmentation": 0.25, "weight_vibration": 0.25}
    selected = select_best_design(df_pareto, weights)

    explanation = generate_trade_off_explanation(df_pareto, selected)

    assert isinstance(explanation, str)
    assert len(explanation.strip()) > 0
    assert "ground vibration" in explanation.lower() or "fragmentation" in explanation.lower()
    assert len(explanation.split()) <= 150
