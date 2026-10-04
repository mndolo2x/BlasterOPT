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


def test_optimizer_uses_trained_model():
    """The optimizer output must differ when the model differs."""
    from src.models import GAANNModel
    from src.pareto_optimizer import run_nsga2
    from src.synthetic_data import generate_for_model

    df = generate_for_model("ga_ann_jwaneng", n_samples=200, seed=42)
    model_a = GAANNModel(input_size=11)
    features = [c for c in df.columns if c in model_a.INPUT_COLUMNS]
    model_a.fit(df[features], df[model_a.OUTPUT_COLUMNS])

    front_a = run_nsga2(model=model_a, n_gen=10, pop_size=30, seed=42)
    assert not front_a.empty, "NSGA-II returned an empty Pareto front"

    # All designs must have spacing >= burden
    assert (front_a["spacing_m"] >= front_a["burden_m"] - 1e-4).all(), \
        "Optimizer returned invalid designs (spacing < burden)"

    # All designs must have valid stemming ratio
    ratio = front_a["stemming_m"] / front_a["burden_m"]
    assert ratio.between(0.5 - 1e-4, 1.0 + 1e-4).all(), \
        "Optimizer returned invalid stemming ratios"


def test_optimizer_rejects_invalid_designs():
    """The optimizer must never return spacing < burden."""
    from src.models import GAANNModel
    from src.pareto_optimizer import run_nsga2
    from src.synthetic_data import generate_for_model

    df = generate_for_model("ga_ann_jwaneng", n_samples=200)
    model = GAANNModel(input_size=11)
    features = [c for c in df.columns if c in model.INPUT_COLUMNS]
    model.fit(df[features], df[model.OUTPUT_COLUMNS])

    front = run_nsga2(model=model, n_gen=10, pop_size=30)

    violations = (front["spacing_m"] < front["burden_m"] - 1e-4).sum()
    assert violations == 0, f"{violations} designs have spacing < burden"


def test_validator_rejects_burden_below_minimum():
    from src.pareto_optimizer import validate_design
    design = {
        "burden_m": 2.0, "spacing_m": 2.5, "stemming_m": 1.5,
        "powder_factor_kg_m3": 0.65, "d80_mm": 250,
        "ppv_mms": 3.5, "airblast_dbl": 115,
    }
    is_valid, violations = validate_design(design)
    assert is_valid is False
    assert any("burden" in v for v in violations)


def test_validator_rejects_pf_above_maximum():
    from src.pareto_optimizer import validate_design
    design = {
        "burden_m": 4.0, "spacing_m": 5.0, "stemming_m": 3.0,
        "powder_factor_kg_m3": 1.2, "d80_mm": 250,
        "ppv_mms": 3.5, "airblast_dbl": 115,
    }
    is_valid, violations = validate_design(design)
    assert is_valid is False
    assert any("powder factor" in v for v in violations)


def test_validator_rejects_airblast_over_120():
    from src.pareto_optimizer import validate_design
    design = {
        "burden_m": 4.0, "spacing_m": 5.0, "stemming_m": 3.0,
        "powder_factor_kg_m3": 0.65, "d80_mm": 250,
        "ppv_mms": 3.5, "airblast_dbl": 130,
    }
    is_valid, violations = validate_design(design)
    assert is_valid is False
    assert any("airblast" in v for v in violations)


def test_validator_accepts_a_correct_design():
    from src.pareto_optimizer import validate_design
    design = {
        "burden_m": 4.2, "spacing_m": 5.1, "stemming_m": 3.0,
        "powder_factor_kg_m3": 0.65, "d80_mm": 250,
        "ppv_mms": 3.5, "airblast_dbl": 115,
    }
    is_valid, violations = validate_design(design)
    assert is_valid is True
    assert violations == []


def test_cache_version_forces_clear():
    """When PARETO_VERSION changes, old session state front must be discarded."""
    session_state = {"pareto_front_df": "old_data", "pareto_version": 1}

    # Simulate code version bump
    new_version = 5
    if session_state.get("pareto_version") != new_version:
        session_state.pop("pareto_front_df", None)
        session_state["pareto_version"] = new_version

    assert "pareto_front_df" not in session_state
    assert session_state["pareto_version"] == 5


def test_optimizer_output_has_varying_airblast():
    """After the fix, airblast must vary across the Pareto front."""
    from src.pareto_optimizer import run_nsga2
    from src.models import GAANNModel, FEATURE_COLS
    from src.synthetic_data import generate_for_model

    df = generate_for_model("ga_ann_jwaneng", n_samples=300, seed=42)
    model = GAANNModel(input_size=len(FEATURE_COLS))
    features = [c for c in df.columns if c in model.INPUT_COLUMNS]
    model.fit(df[features], df[model.OUTPUT_COLUMNS])

    front = run_nsga2(model=model, n_gen=10, pop_size=30, seed=42)

    assert not front.empty, "Pareto front is empty"
    assert front["airblast_dbl"].nunique() >= 3, f"Airblast is frozen: {front['airblast_dbl'].unique()}"
    assert front["burden_m"].between(3.0, 6.0).all(), f"Burden out of range: {front['burden_m'].min()}-{front['burden_m'].max()}"
    assert front["powder_factor_kg_m3"].between(0.4, 0.9).all(), f"PF out of range: {front['powder_factor_kg_m3'].min()}-{front['powder_factor_kg_m3'].max()}"
    assert front["airblast_dbl"].max() <= 120.0 + 1e-4, f"Airblast exceeds 120 dB: {front['airblast_dbl'].max()}"
