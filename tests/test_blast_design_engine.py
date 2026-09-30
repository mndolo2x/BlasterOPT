"""
Unit tests for BlastDesignEngine in src/fragmentation/blast_design_engine.py.
"""

import pytest
from src.fragmentation import BlastDesignEngine


class MockInputs:
    def __init__(self, frag_model="kco"):
        self.block_id = "BLOCK_JWA_01"
        self.fragmentation_model = frag_model
        self.burden_m = 4.0
        self.spacing_m = 5.0
        self.bench_height_m = 12.0
        self.hole_diameter_mm = 250.0
        self.powder_factor_kg_m3 = 0.65
        self.explosive_rws = 100.0


def test_blast_design_engine_kco():
    engine = BlastDesignEngine()
    inputs = MockInputs(frag_model="kco")
    res = engine.design(inputs)

    assert "block_id" in res
    assert "rock_factor_a" in res
    assert "fragmentation_prediction" in res
    assert res["fragmentation_prediction"]["model"] == "KCO"
    assert "x_50_cm" in res["fragmentation_prediction"]


def test_blast_design_engine_kuz_ram():
    engine = BlastDesignEngine()
    inputs = MockInputs(frag_model="kuz_ram")
    res = engine.design(inputs)

    assert res["fragmentation_prediction"]["model"] == "Kuz-Ram"
    assert "d50_mm" in res["fragmentation_prediction"]
