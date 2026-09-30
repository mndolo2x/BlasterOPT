"""
Unit tests for BlastDesignEngine.
"""

from unittest.mock import MagicMock
from src.fragmentation.blast_design_engine import BlastDesignEngine


def test_blast_design_engine_integration():
    engine = BlastDesignEngine()
    inputs = MagicMock()
    inputs.block_id = "BLOCK_001"
    inputs.fragmentation_model = "kco"
    inputs.burden_m = 3.5
    inputs.spacing_m = 4.5
    inputs.bench_height_m = 12.0
    inputs.powder_factor_kg_m3 = 0.65
    inputs.explosive_rws = 100.0
    inputs.rock_factor_a = 8.0
    inputs.hole_diameter_mm = 200.0
    inputs.explosive_type = "ANFO"
    inputs.stemming_m = 2.5
    inputs.max_charge_per_delay_kg = 150.0
    inputs.explosive_mass_kg = 800.0
    inputs.closest_receptor_dist_m = 400.0

    res = engine.design(inputs)

    assert res["block_id"] == "BLOCK_001"
    assert "fragmentation_prediction" in res
    assert "dust_assessment" in res
    assert "gas_assessment" in res
    assert "noise_assessment" in res
    assert "environmental_impact" in res
    assert "risk_assessment" in res
    assert isinstance(res["warnings"], list)
