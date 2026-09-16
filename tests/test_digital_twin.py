"""
Unit tests for Mine-to-Mill Digital Twin module (src/digital_twin.py).
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
from src.digital_twin import (
    FragmentationModel,
    DownstreamModel,
    MineToMillTwin,
    OreTracker,
    ScenarioAnalyzer,
)


def test_fragmentation_model_predict_returns_valid_distribution():
    """Test FragmentationModel.predict() returns a valid DataFrame with size_cm and percent_passing columns."""
    model = FragmentationModel(rock_factor_A=8.5)
    params = {"powder_factor_kg_m3": 0.65, "charge_mass_per_hole_kg": 320.0}

    df_dist = model.predict(params)

    assert isinstance(df_dist, pd.DataFrame)
    assert not df_dist.empty
    assert "size_cm" in df_dist.columns
    assert "percent_passing" in df_dist.columns

    # Test get_percentile
    d50 = model.get_percentile(50.0)
    assert isinstance(d50, float)
    assert d50 > 0.0


def test_mine_to_mill_twin_simulate_returns_dict_with_required_keys():
    """Test MineToMillTwin.simulate() returns a dict with fragmentation, throughput, energy, and cost keys."""
    twin = MineToMillTwin(rock_factor_A=8.5, ore_hardness_wi=12.5)
    params = {
        "powder_factor_kg_m3": 0.65,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "bench_height_m": 15.0,
        "charge_mass_per_hole_kg": 320.0,
    }

    res = twin.simulate(params)

    assert isinstance(res, dict)
    assert "fragmentation" in res
    assert "throughput" in res
    assert "energy" in res
    assert "cost" in res

    assert res["throughput"] > 0.0
    assert res["energy"] > 0.0
    assert res["cost"] > 0.0


def test_scenario_analyzer_sensitivity_sweep_returns_plotly_figure():
    """Test ScenarioAnalyzer.sensitivity_sweep() returns an interactive Plotly Figure."""
    analyzer = ScenarioAnalyzer()
    params = {"powder_factor_kg_m3": 0.65, "burden_m": 6.0, "spacing_m": 7.0}

    fig = analyzer.sensitivity_sweep(params, param_to_vary="powder_factor_kg_m3", range_min=0.30, range_max=1.20)

    assert fig is not None
    assert hasattr(fig, "data")


def test_ore_tracker_trace_ore_returns_list_of_nodes():
    """Test OreTracker.trace_ore() returns a list of nodes tracing path from bench to mill."""
    tracker = OreTracker()
    tracker.add_blast("BLAST_TEST_01", coordinates=(-24.5, 25.8, 1150.0), timestamp="2026-09-16T10:00:00")
    tracker.add_ore_block("BLOCK_TEST_01", blast_id="BLAST_TEST_01", coordinates=(-24.51, 25.81, 1150.0))
    tracker.add_processing_batch("BATCH_TEST_01", block_ids=["BLOCK_TEST_01"], timestamp="2026-09-16T14:00:00")

    path = tracker.trace_ore("BLOCK_TEST_01")

    assert isinstance(path, list)
    assert len(path) == 3
    assert path[0]["node_type"] == "Blast"
    assert path[1]["node_type"] == "OreBlock"
    assert path[2]["node_type"] == "ProcessingBatch"
