"""
Unit tests for the 3D Mine-to-Mill Digital Twin module (Model 4).
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.digital_twin import (
    build_digital_twin,
    simulate_fragmentation,
    link_to_downstream,
    FragmentationModel,
    DownstreamModel,
    MineToMillTwin,
    OreTracker,
    ScenarioAnalyzer,
)


def test_fragmentation_model_predict_returns_valid_distribution():
    """Test FragmentationModel.predict returns a valid distribution dictionary with required percentiles."""
    frag_model = FragmentationModel(rock_factor_A=8.5)
    blast_params = {
        "powder_factor_kg_m3": 0.65,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "charge_mass_per_hole_kg": 320.0,
    }

    res = frag_model.predict(blast_params)

    assert isinstance(res, dict)
    assert "d10_mm" in res
    assert "d50_mm" in res
    assert "d80_mm" in res
    assert "d90_mm" in res
    assert "characteristic_size_xc_mm" in res
    assert "uniformity_index_n" in res
    assert "boulder_percentage" in res
    assert "fines_percentage" in res

    # Monotonic size progression check: d10 < d50 < d80 < d90
    assert res["d10_mm"] < res["d50_mm"] < res["d80_mm"] < res["d90_mm"]


def test_mine_to_mill_twin_simulate_returns_dict_with_required_keys():
    """Test MineToMillTwin.simulate returns a dict with fragmentation, throughput, energy, and cost keys."""
    twin = MineToMillTwin(rock_factor_A=8.5, ore_hardness_wi=12.5)
    blast_params = {
        "powder_factor_kg_m3": 0.70,
        "burden_m": 5.5,
        "spacing_m": 6.5,
        "stemming_m": 4.5,
        "bench_height_m": 15.0,
        "hole_diameter_mm": 250.0,
    }

    res = twin.simulate(blast_params)

    assert isinstance(res, dict)
    assert "fragmentation" in res
    assert "throughput" in res
    assert "energy" in res
    assert "cost" in res
    assert "downstream" in res

    assert res["throughput"] > 0
    assert res["energy"] > 0
    assert res["cost"] > 0


def test_mine_to_mill_twin_what_if_dataframe():
    """Test MineToMillTwin.what_if generates a sensitivity DataFrame over parameter range."""
    twin = MineToMillTwin()
    blast_params = {"powder_factor_kg_m3": 0.65, "burden_m": 6.0, "spacing_m": 7.0}

    df_sweep = twin.what_if(blast_params, param_to_vary="powder_factor_kg_m3", range_min=0.30, range_max=1.00, steps=5)

    assert isinstance(df_sweep, pd.DataFrame)
    assert len(df_sweep) == 5
    assert "powder_factor_kg_m3" in df_sweep.columns
    assert "total_cost_per_tonne_usd" in df_sweep.columns
    assert "crusher_throughput_tph" in df_sweep.columns


def test_scenario_analyzer_returns_plotly_figure():
    """Test ScenarioAnalyzer.plot_sensitivity returns a valid Plotly Figure."""
    analyzer = ScenarioAnalyzer()
    blast_params = {"powder_factor_kg_m3": 0.65, "burden_m": 6.0, "spacing_m": 7.0}

    fig = analyzer.plot_sensitivity(blast_params, param_to_vary="powder_factor_kg_m3", range_min=0.30, range_max=1.00)

    assert fig is not None
    assert isinstance(fig, go.Figure)
    assert hasattr(fig, "data")


def test_ore_tracker_records_graph_relationship():
    """Test OreTracker records ore relationships from bench to mill."""
    tracker = OreTracker()
    record = tracker.track_ore_block(blast_id="BLAST_TEST_01", block_id="BLOCK_TEST_01")

    assert isinstance(record, dict)
    assert record["blast_id"] == "BLAST_TEST_01"
    assert record["status"] == "TRACKED_BENCH_TO_MILL"
    tracker.close()


def test_build_digital_twin_standalone():
    """Test build_digital_twin returns valid 3D bench block geometry structure."""
    dt = build_digital_twin(bench_id="BENCH_TEST_01")

    assert isinstance(dt, dict)
    assert dt["bench_id"] == "BENCH_TEST_01"
    assert "dimensions" in dt
    assert "vertices_3d" in dt
