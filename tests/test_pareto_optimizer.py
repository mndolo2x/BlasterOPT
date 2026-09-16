"""
Unit tests for Multi-Objective Pareto Optimizer module (src/pareto_optimizer.py).
"""

import pytest
import pandas as pd
from src.pareto_optimizer import run_nsga2, select_best_design, generate_trade_off_explanation, plot_pareto_front


def test_run_nsga2_returns_dataframe_with_correct_columns():
    """Test run_nsga2 returns a DataFrame containing all required decision variables and objective outcomes."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)

    assert isinstance(df_pareto, pd.DataFrame)
    assert not df_pareto.empty

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
        assert col in df_pareto.columns, f"Missing expected column {col} in Pareto front DataFrame"


def test_select_best_design_returns_row_from_pareto_front():
    """Test select_best_design returns a valid row dictionary from the Pareto front."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)

    weights = {
        "weight_fragmentation": 0.30,
        "weight_vibration": 0.20,
        "weight_airblast": 0.15,
        "weight_cost": 0.20,
        "weight_throughput": 0.15,
    }

    selected = select_best_design(df_pareto, weights)

    assert isinstance(selected, dict)
    assert "burden_m" in selected
    assert "d80_mm" in selected
    assert "ppv_mms" in selected
    assert "utility_score" in selected


def test_generate_trade_off_explanation_returns_non_empty_string():
    """Test generate_trade_off_explanation returns a non-empty descriptive string."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)
    selected = select_best_design(df_pareto, {"weight_vibration": 0.50})

    explanation = generate_trade_off_explanation(df_pareto, selected)

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "vibration" in explanation.lower() or "design" in explanation.lower()


def test_plot_pareto_front_returns_figure():
    """Test plot_pareto_front generates an interactive Plotly Figure."""
    df_pareto = run_nsga2(n_gen=10, pop_size=20, seed=42)
    fig = plot_pareto_front(df_pareto, x_objective="d80_mm", y_objective="ppv_mms")

    assert fig is not None
    assert hasattr(fig, "data")
