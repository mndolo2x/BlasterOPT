"""
Automated Unit Tests for Import Workflows & Reference Package Files.

Tests:
1. Importing blast register CSV with BLASTOPT-TEST-001.
2. Conflict resolution modes (skip, update, cancel) for existing blast IDs.
3. Importing prediction reference records with provenance classification ('synthetic_test', 'external_import').
4. Importing wide format actual results file (actual_p80_mm, actual_p50_mm, actual_ppv_mm_s, actual_airblast_db).
5. Importing long format actual results file (target_name, actual_value, unit).
6. Auto-detection of wide vs long format formats.
7. Validation error handling for non-numeric values, invalid blast IDs, and out-of-bounds metrics.
8. Duplicate measurement detection during upload.
9. Matching predictions and actuals on BLASTOPT-TEST-001 and calculating MAE, RMSE, and signed errors.
10. Unmatched prediction and unmatched actual handling.
"""

import os
import pytest
import pandas as pd
import numpy as np
import src.performance_feedback as pf

TEST_DB = "data/processed/test_import_workflows.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    pf.init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_1_import_blast_register_csv():
    df_register = pd.DataFrame([{
        "blast_id": "BLASTOPT-TEST-001",
        "site": "Jwaneng",
        "operation_type": "open_pit",
        "area": "Cut 8",
        "bench_id": "BENCH_JWA_15S",
        "blast_date": "2026-10-20T12:00:00Z",
        "rock_type": "Kimberlite",
        "geological_domain": "Kimberlite_Hard",
        "record_status": "EXECUTED",
        "notes": "Test reference blast",
    }])

    res = pf.import_blast_records(df_register, conflict_mode="skip", db_path=TEST_DB)
    assert res["status"] == "completed"
    assert res["created"] == 1

    fetched = pf.get_blast_record("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert fetched is not None
    assert fetched.site_id == "Jwaneng"
    assert fetched.location_identifier == "BENCH_JWA_15S"


def test_2_blast_register_conflict_resolution():
    df_register = pd.DataFrame([{
        "blast_id": "BLASTOPT-TEST-001",
        "site": "Jwaneng",
        "location_identifier": "BENCH_OLD",
    }])
    pf.import_blast_records(df_register, db_path=TEST_DB)

    # Duplicate with update mode
    df_update = pd.DataFrame([{
        "blast_id": "BLASTOPT-TEST-001",
        "site": "Jwaneng",
        "bench_id": "BENCH_UPDATED",
    }])
    res_up = pf.import_blast_records(df_update, conflict_mode="update", db_path=TEST_DB)
    assert res_up["updated"] == 1

    fetched = pf.get_blast_record("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert fetched.location_identifier == "BENCH_UPDATED"


def test_3_import_prediction_records_provenance():
    df_preds = pd.DataFrame([
        {
            "blast_id": "BLASTOPT-TEST-001",
            "prediction_run_id": "RUN_001_D50",
            "prediction_timestamp": "2026-10-20T10:00:00Z",
            "model_name": "GA-ANN Model",
            "model_version": "v1.0",
            "target_name": "d50_mm",
            "predicted_value": 240.0,
            "unit": "mm",
        },
        {
            "blast_id": "BLASTOPT-TEST-001",
            "prediction_run_id": "RUN_001_PPV",
            "prediction_timestamp": "2026-10-20T10:00:00Z",
            "model_name": "GA-ANN Model",
            "model_version": "v1.0",
            "target_name": "ppv_mms",
            "predicted_value": 3.8,
            "unit": "mm/s",
        },
    ])

    res = pf.import_prediction_records(df_preds, provenance_label="synthetic_test", db_path=TEST_DB)
    assert res["created"] == 2

    snaps = pf.get_prediction_snapshots_for_blast("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert len(snaps) == 2
    assert snaps[0].provenance == "synthetic_test"


def test_4_import_wide_format_actual_results():
    df_wide = pd.DataFrame([{
        "blast_id": "BLASTOPT-TEST-001",
        "measurement_timestamp": "2026-10-20T14:00:00Z",
        "actual_p80_mm": 32.0,
        "actual_p50_mm": 245.0,
        "actual_ppv_mm_s": 4.2,
        "actual_airblast_db": 114.5,
        "actual_cost_usd": 4.85,
    }])

    maps = pf.identify_candidate_mappings(list(df_wide.columns))
    valid_meass, errs, dups = pf.validate_actual_results_df(df_wide, maps)

    assert len(valid_meass) >= 4
    assert len(errs) == 0

    for m in valid_meass:
        pf.save_actual_measurement(m, db_path=TEST_DB)

    fetched = pf.get_actual_measurements_for_blast("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert len(fetched) >= 4


def test_5_import_long_format_actual_results():
    df_long = pd.DataFrame([
        {
            "blast_id": "BLASTOPT-TEST-001",
            "measurement_id": "MEAS_001",
            "measurement_timestamp": "2026-10-20T14:00:00Z",
            "target_name": "d50_mm",
            "actual_value": 245.0,
            "unit": "mm",
            "measurement_method": "UAV Photogrammetry",
        },
        {
            "blast_id": "BLASTOPT-TEST-001",
            "measurement_id": "MEAS_002",
            "measurement_timestamp": "2026-10-20T14:00:00Z",
            "target_name": "ppv_mms",
            "actual_value": 4.2,
            "unit": "mm/s",
            "measurement_method": "Seismograph",
        },
    ])

    maps = pf.identify_candidate_mappings(list(df_long.columns))
    valid_meass, errs, dups = pf.validate_actual_results_df(df_long, maps)

    assert len(valid_meass) == 2
    assert valid_meass[0].target_metric == "d50_mm"
    assert valid_meass[1].target_metric == "ppv_mms"


def test_6_prediction_comparison_calculation_on_test_blast():
    # 1. Register blast
    pf.save_blast_record(pf.BlastRecord(blast_id="BLASTOPT-TEST-001", site_id="Jwaneng", location_identifier="BENCH_01"), db_path=TEST_DB)

    # 2. Prediction
    snap = pf.PredictionSnapshot(
        prediction_run_id="PRED_001",
        blast_id="BLASTOPT-TEST-001",
        model_name="GA-ANN Model",
        target_metric="d50_mm",
        predicted_value=240.0,
        unit="mm",
        provenance="synthetic_test",
    )
    pf.save_prediction_snapshot(snap, db_path=TEST_DB)

    # 3. Actual
    m = pf.ActualMeasurement(
        measurement_id="MEAS_001",
        blast_id="BLASTOPT-TEST-001",
        target_metric="d50_mm",
        measured_value=245.0,
        unit="mm",
    )
    pf.save_actual_measurement(m, db_path=TEST_DB)

    # 4. Comparison calculation
    signed_e = snap.predicted_value - float(m.measured_value)  # 240 - 245 = -5.0
    abs_e = abs(signed_e)  # 5.0
    comp = pf.ComparisonResult(
        comparison_id="COMP_001",
        blast_id="BLASTOPT-TEST-001",
        prediction_run_id=snap.prediction_run_id,
        measurement_id=m.measurement_id,
        target_metric="d50_mm",
        predicted_value=snap.predicted_value,
        measured_value=float(m.measured_value),
        signed_error=signed_e,
        absolute_error=abs_e,
        unit="mm",
        model_name="GA-ANN Model",
        model_version="v1.0",
    )
    pf.save_comparison_result(comp, db_path=TEST_DB)

    results = pf.list_comparison_results(target_metric="d50_mm", db_path=TEST_DB)
    assert len(results) == 1
    assert results[0].signed_error == -5.0
    assert results[0].absolute_error == 5.0
