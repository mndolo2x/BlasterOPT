"""
Unit tests for the 3D Digital Twin of the Bench module.
"""

import pytest
import pandas as pd
import numpy as np
from src.digital_twin import build_digital_twin, simulate_fragmentation, link_to_downstream


def test_build_digital_twin_returns_valid_structure():
    """Test build_digital_twin returns a 3D model dictionary with expected keys."""
    dt = build_digital_twin(bench_id="BENCH_TEST_01")

    assert isinstance(dt, dict)
    assert dt["bench_id"] == "BENCH_TEST_01"
    assert "geological_data" in dt
    assert "as_drilled_data" in dt
    assert "dimensions" in dt
    assert "vertices_3d" in dt
    assert isinstance(dt["as_drilled_data"], pd.DataFrame)
    assert isinstance(dt["vertices_3d"], np.ndarray)
    assert dt["vertices_3d"].shape == (8, 3)


def test_simulate_fragmentation_returns_distribution_sizes():
    """Test simulate_fragmentation returns d10, d50, d80, d90 and Rosin-Rammler parameters."""
    dt = build_digital_twin(bench_id="BENCH_TEST_02")
    blast_params = {
        "powder_factor_kg_m3": 0.70,
        "burden_m": 5.5,
        "spacing_m": 6.5,
        "stemming_m": 4.5,
        "charge_mass_per_hole_kg": 350.0,
    }

    frag = simulate_fragmentation(dt, blast_params=blast_params)

    assert isinstance(frag, dict)
    assert "d10_mm" in frag
    assert "d50_mm" in frag
    assert "d80_mm" in frag
    assert "d90_mm" in frag
    assert "characteristic_size_xc_mm" in frag
    assert "uniformity_index_n" in frag
    assert "boulder_percentage" in frag
    assert "fines_percentage" in frag

    # Check size progression: d10 < d50 < d80 < d90
    assert frag["d10_mm"] < frag["d50_mm"] < frag["d80_mm"] < frag["d90_mm"]


def test_link_to_downstream_outcomes():
    """Test link_to_downstream predicts digger, truck, crusher, and cost KPIs."""
    dt = build_digital_twin(bench_id="BENCH_TEST_03")
    frag = simulate_fragmentation(dt)

    downstream = link_to_downstream(dt, frag)

    assert isinstance(downstream, dict)
    assert "digger_productivity_tph" in downstream
    assert "bucket_fill_factor_pct" in downstream
    assert "truck_fill_factor_pct" in downstream
    assert "crusher_throughput_tph" in downstream
    assert "specific_energy_kwh_t" in downstream
    assert "total_cost_per_tonne_usd" in downstream

    assert downstream["digger_productivity_tph"] > 0
    assert downstream["crusher_throughput_tph"] > 0
    assert downstream["total_cost_per_tonne_usd"] > 0
