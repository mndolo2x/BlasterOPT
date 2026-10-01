"""
Unit tests for FlyrockPredictor in src/models.py.
"""
import pytest
import pandas as pd
from src.models import FlyrockPredictor


def test_flyrock_predicts_one_output():
    assert FlyrockPredictor.OUTPUT_COLUMNS == ["flyrock_m"]


def test_flyrock_increases_with_charge():
    m = FlyrockPredictor()
    df_low = pd.DataFrame([{"charge_mass_per_hole_kg": 200.0, "stemming_m": 5.0, "burden_m": 6.0}])
    df_high = pd.DataFrame([{"charge_mass_per_hole_kg": 500.0, "stemming_m": 5.0, "burden_m": 6.0}])

    pred_low = m.predict(df_low)["flyrock_m"].values[0]
    pred_high = m.predict(df_high)["flyrock_m"].values[0]

    assert pred_high > pred_low, f"Expected higher flyrock with higher charge: {pred_high} > {pred_low}"


def test_flyrock_decreases_with_stemming():
    m = FlyrockPredictor()
    df_low_stem = pd.DataFrame([{"charge_mass_per_hole_kg": 300.0, "stemming_m": 3.0, "burden_m": 6.0}])
    df_high_stem = pd.DataFrame([{"charge_mass_per_hole_kg": 300.0, "stemming_m": 7.0, "burden_m": 6.0}])

    pred_low = m.predict(df_low_stem)["flyrock_m"].values[0]
    pred_high = m.predict(df_high_stem)["flyrock_m"].values[0]

    assert pred_high < pred_low, f"Expected lower flyrock with higher stemming: {pred_high} < {pred_low}"


def test_flyrock_decreases_with_burden():
    m = FlyrockPredictor()
    df_low_burden = pd.DataFrame([{"charge_mass_per_hole_kg": 300.0, "stemming_m": 5.0, "burden_m": 4.0}])
    df_high_burden = pd.DataFrame([{"charge_mass_per_hole_kg": 300.0, "stemming_m": 5.0, "burden_m": 8.0}])

    pred_low = m.predict(df_low_burden)["flyrock_m"].values[0]
    pred_high = m.predict(df_high_burden)["flyrock_m"].values[0]

    assert pred_high < pred_low, f"Expected lower flyrock with higher burden: {pred_high} < {pred_low}"


def test_flyrock_within_clip_range():
    m = FlyrockPredictor()
    df = pd.DataFrame([
        {"charge_mass_per_hole_kg": 1.0, "stemming_m": 10.0, "burden_m": 2.0},
        {"charge_mass_per_hole_kg": 2000.0, "stemming_m": 1.0, "burden_m": 2.0},
    ])

    preds = m.predict(df)["flyrock_m"]
    assert (preds >= 5.0).all() and (preds <= 500.0).all()
