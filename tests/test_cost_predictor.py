"""
Unit tests for CostPredictor in src/models.py.
"""
import pytest
import pandas as pd
from src.models import CostPredictor


def test_cost_predicts_one_output():
    assert CostPredictor.OUTPUT_COLUMNS == ["cost_per_tonne_usd"]


def test_cost_increases_with_hole_depth():
    m = CostPredictor()
    df_low = pd.DataFrame([{"hole_depth_m": 10.0, "charge_mass_per_hole_kg": 320.0}])
    df_high = pd.DataFrame([{"hole_depth_m": 20.0, "charge_mass_per_hole_kg": 320.0}])

    cost_low = m.predict(df_low)["cost_per_tonne_usd"].values[0]
    cost_high = m.predict(df_high)["cost_per_tonne_usd"].values[0]

    assert cost_high > cost_low, f"Expected higher cost with deeper holes: {cost_high} > {cost_low}"


def test_cost_increases_with_charge():
    m = CostPredictor()
    df_low = pd.DataFrame([{"hole_depth_m": 15.0, "charge_mass_per_hole_kg": 200.0}])
    df_high = pd.DataFrame([{"hole_depth_m": 15.0, "charge_mass_per_hole_kg": 500.0}])

    cost_low = m.predict(df_low)["cost_per_tonne_usd"].values[0]
    cost_high = m.predict(df_high)["cost_per_tonne_usd"].values[0]

    assert cost_high > cost_low, f"Expected higher cost with higher charge: {cost_high} > {cost_low}"


def test_cost_within_reasonable_range():
    m = CostPredictor()
    df_typical = pd.DataFrame([{"hole_depth_m": 16.5, "charge_mass_per_hole_kg": 320.0}])
    cost = m.predict(df_typical)["cost_per_tonne_usd"].values[0]

    assert 0.50 <= cost <= 3.00, f"Expected cost between 0.50 and 3.00 USD/t, got {cost}"
    assert abs(cost - 0.76) < 0.05, f"Expected typical cost ~$0.76/t, got {cost}"
