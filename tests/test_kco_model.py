"""
Unit tests for KCOModel in src/fragmentation/kco.py.
"""

import pytest
import pandas as pd
from src.fragmentation import KCOModel


def test_kco_model_predict():
    params = {
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "powder_factor_kg_m3": 0.65,
        "explosive_rws": 100.0,
        "rock_factor_a": 8.0,
        "blastability_index": 50.0,
        "in_situ_block_size_m": 0.5,
    }

    kco = KCOModel()
    res = kco.predict(params)

    assert res["model"] == "KCO"
    assert res["x_50_cm"] > 0.0
    assert res["x_max_cm"] > res["x_50_cm"]
    assert "distribution" in res
    assert isinstance(res["distribution"], pd.DataFrame)


def test_kco_compare_with_kuz_ram():
    params = {
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "powder_factor_kg_m3": 0.65,
        "explosive_rws": 100.0,
        "rock_factor_a": 8.0,
    }

    kco = KCOModel()
    comp_df = kco.compare_with_kuz_ram(params)

    assert isinstance(comp_df, pd.DataFrame)
    assert "size_mm" in comp_df.columns
    assert "kuz_ram_passing" in comp_df.columns
    assert "kco_passing" in comp_df.columns
