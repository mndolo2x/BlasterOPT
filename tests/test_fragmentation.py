"""
Unit tests for src/fragmentation.py.
"""
import pytest
from src.fragmentation import KuzRamModel, SwebrecModel, KCOModel, OpticalImageAnalyzer, RockFactorCalibrator, BlastDesignEngine


def test_kuz_ram_model():
    res = KuzRamModel.calculate_distribution()
    assert res["d50_mm"] > 0.0
    assert res["d80_mm"] > res["d50_mm"]
    assert not res["distribution_df"].empty


def test_swebrec_model():
    df_swe = SwebrecModel.calculate_swebrec(x50_cm=20.0)
    assert not df_swe.empty
    assert "size_cm" in df_swe.columns
    assert "percent_passing" in df_swe.columns


def test_kco_model():
    df_kco = KCOModel.calculate_kco(x50_cm=20.0, n_val=1.2)
    assert not df_kco.empty


def test_optical_analyzer():
    res = OpticalImageAnalyzer.import_wipfrag_data()
    assert "d50_mm" in res
    assert res["d50_mm"] > 0.0


def test_rock_factor_calibrator():
    cal_a = RockFactorCalibrator.calibrate_rock_factor(measured_d50_mm=240.0, predicted_d50_mm=200.0, current_A=8.0)
    assert 4.0 <= cal_a <= 16.0


def test_blast_design_engine():
    res = BlastDesignEngine.evaluate_design({"powder_factor_kg_m3": 0.65})
    assert "is_feasible" in res
    assert res["is_feasible"] is True
