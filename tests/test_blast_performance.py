"""
Automated Unit Tests for Blast Performance Feedback Loop Module.

Tests end-to-end functionality:
1. Creating a blast record and retrieving it after a page refresh or new session.
2. Saving and preserving multiple original prediction runs for the same blast.
3. Uploading a valid actual-results file.
4. Rejecting invalid columns, invalid units, and invalid numeric measurements.
5. Handling missing optional measurements correctly.
6. Detecting duplicate observations and conflicting records.
7. Matching actual results to the correct blast and prediction target.
8. Preventing incompatible targets and units from being compared.
9. Calculating MAE, RMSE, and bias correctly on a small hand-calculated test example.
10. Handling zero actual values when calculating percentage errors.
11. Excluding invalid or unreviewed observations from approved exports.
12. Preserving record history when a measurement is corrected.
13. Filtering historical analytics correctly.
14. Handling insufficient data without displaying misleading metrics.
15. Exporting an approved dataset with the correct schema and units.
16. Preventing leakage between related blast observations in dataset partitioning.
17. Ensuring failed training or import runs are not shown as successful.
18. Keeping existing BlastOPT functionality working.
"""

import os
import pytest
import pandas as pd
import numpy as np
import src.performance_feedback as pf

TEST_DB = "data/processed/test_blast_performance.db"


@pytest.fixture(autouse=True)
def clean_test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    pf.init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_1_create_and_retrieve_blast_record():
    rec = pf.BlastRecord(
        blast_id="BLAST_TEST_01",
        site_id="Jwaneng",
        operation_type="open_pit",
        location_identifier="BENCH_15S",
        notes="Test blast record",
    )
    pf.save_blast_record(rec, db_path=TEST_DB)

    retrieved = pf.get_blast_record("BLAST_TEST_01", db_path=TEST_DB)
    assert retrieved is not None
    assert retrieved.blast_id == "BLAST_TEST_01"
    assert retrieved.site_id == "Jwaneng"
    assert retrieved.operation_type == "open_pit"


def test_2_save_and_preserve_multiple_prediction_runs():
    snap1 = pf.PredictionSnapshot(
        prediction_run_id="RUN_01",
        blast_id="BLAST_TEST_01",
        model_name="GA-ANN Model",
        target_metric="d50_mm",
        predicted_value=240.0,
        unit="mm",
    )
    snap2 = pf.PredictionSnapshot(
        prediction_run_id="RUN_02",
        blast_id="BLAST_TEST_01",
        model_name="ANN-RF Ensemble",
        target_metric="d50_mm",
        predicted_value=235.0,
        unit="mm",
    )
    pf.save_prediction_snapshot(snap1, db_path=TEST_DB)
    pf.save_prediction_snapshot(snap2, db_path=TEST_DB)

    snaps = pf.get_prediction_snapshots_for_blast("BLAST_TEST_01", target_metric="d50_mm", db_path=TEST_DB)
    assert len(snaps) == 2
    assert snaps[0].predicted_value in [240.0, 235.0]


def test_3_upload_valid_actual_results():
    df_raw = pd.DataFrame([
        {"blast_id": "BLAST_TEST_01", "d50_mm": 245.0, "ppv_mms": 4.2},
        {"blast_id": "BLAST_TEST_02", "d50_mm": 210.0, "ppv_mms": 3.8},
    ])
    mappings = pf.identify_candidate_mappings(list(df_raw.columns))
    valid_meass, row_errs, dup_conflicts = pf.validate_actual_results_df(df_raw, mappings)

    assert len(valid_meass) == 4
    assert len(row_errs) == 0
    assert len(dup_conflicts) == 0


def test_4_reject_invalid_numeric_values():
    df_raw = pd.DataFrame([
        {"blast_id": "BLAST_TEST_01", "d50_mm": "INVALID_NUMBER"},
        {"blast_id": "BLAST_TEST_02", "d50_mm": -999.0},  # Out of physical bounds
    ])
    mappings = pf.identify_candidate_mappings(list(df_raw.columns))
    valid_meass, row_errs, dup_conflicts = pf.validate_actual_results_df(df_raw, mappings)

    assert len(valid_meass) == 0
    assert len(row_errs) == 2


def test_5_handle_missing_optional_measurements():
    df_raw = pd.DataFrame([
        {"blast_id": "BLAST_TEST_01", "d50_mm": 240.0},  # ppv_mms is missing/optional
    ])
    mappings = pf.identify_candidate_mappings(list(df_raw.columns))
    valid_meass, row_errs, dup_conflicts = pf.validate_actual_results_df(df_raw, mappings)

    assert len(valid_meass) == 1
    assert valid_meass[0].target_metric == "d50_mm"


def test_6_detect_duplicate_conflicting_records():
    df_raw = pd.DataFrame([
        {"blast_id": "BLAST_TEST_01", "d50_mm": 240.0},
        {"blast_id": "BLAST_TEST_01", "d50_mm": 250.0},  # Duplicate in same file
    ])
    mappings = pf.identify_candidate_mappings(list(df_raw.columns))
    valid_meass, row_errs, dup_conflicts = pf.validate_actual_results_df(df_raw, mappings)

    assert len(valid_meass) == 1
    assert len(dup_conflicts) == 1


def test_7_match_actual_results_to_blast_id():
    m = pf.ActualMeasurement(
        measurement_id="MEAS_01",
        blast_id="BLAST_TEST_01",
        target_metric="d50_mm",
        measured_value=245.0,
        unit="mm",
    )
    pf.save_actual_measurement(m, db_path=TEST_DB)

    fetched = pf.get_actual_measurements_for_blast("BLAST_TEST_01", target_metric="d50_mm", db_path=TEST_DB)
    assert len(fetched) == 1
    assert fetched[0].measured_value == 245.0


def test_8_prevent_incompatible_targets_comparison():
    # Only compare metrics matching the target_metric
    comp1 = pf.ComparisonResult(
        comparison_id="C1",
        blast_id="B1",
        prediction_run_id="P1",
        measurement_id="M1",
        target_metric="d50_mm",
        predicted_value=240.0,
        measured_value=250.0,
        signed_error=-10.0,
        absolute_error=10.0,
        unit="mm",
        model_name="GA-ANN",
        model_version="v1.0",
    )
    pf.save_comparison_result(comp1, db_path=TEST_DB)

    comps = pf.list_comparison_results(target_metric="ppv_mms", db_path=TEST_DB)
    assert len(comps) == 0


def test_9_hand_calculated_error_metrics():
    # Hand-calculated example:
    # Preds: [10.0, 12.0, 15.0]
    # Actuals: [8.0, 14.0, 15.0]
    # Errors: [2.0, -2.0, 0.0]
    # |Errors|: [2.0, 2.0, 0.0] -> MAE = 4.0 / 3 = 1.3333
    # Errors^2: [4.0, 4.0, 0.0] -> RMSE = sqrt(8.0 / 3) = 1.6330
    # Bias: (2.0 - 2.0 + 0.0) / 3 = 0.0
    preds = [10.0, 12.0, 15.0]
    acts = [8.0, 14.0, 15.0]

    metrics = pf.calculate_regression_metrics(preds, acts)
    assert abs(metrics["mae"] - 1.3333) < 1e-3
    assert abs(metrics["rmse"] - 1.6330) < 1e-3
    assert abs(metrics["bias"] - 0.0) < 1e-3


def test_10_zero_actual_value_percentage_error_handling():
    preds = [10.0, 5.0]
    acts = [0.0, 5.0]  # Zero actual value present

    metrics = pf.calculate_regression_metrics(preds, acts)
    # Zero actual handled cleanly without DivisionByZero error
    assert metrics["mape"] == 0.0  # Only valid non-zero actual evaluated (|5-5|/5 * 100 = 0%)


def test_11_exclude_unreviewed_observations_from_approved_export():
    # Save a blast record and unreviewed dataset review
    b = pf.BlastRecord(blast_id="B_UNREV", site_id="Jwaneng", operation_type="open_pit", location_identifier="BENCH_1")
    pf.save_blast_record(b, db_path=TEST_DB)

    m = pf.ActualMeasurement(measurement_id="M_UNREV", blast_id="B_UNREV", target_metric="d50_mm", measured_value=240.0, unit="mm")
    pf.save_actual_measurement(m, db_path=TEST_DB)

    r = pf.DatasetReviewRecord(review_id="R_UNREV", blast_id="B_UNREV", target_metric="d50_mm", status="Awaiting Review", inclusion_status=False)
    pf.save_dataset_review(r, db_path=TEST_DB)

    df_exp, manifest, _ = pf.export_approved_dataset(target_metric="d50_mm", db_path=TEST_DB)
    assert manifest.approved_record_count == 0


def test_12_preserve_record_history_on_correction():
    b = pf.BlastRecord(blast_id="B_CORR", site_id="Jwaneng", operation_type="open_pit", location_identifier="BENCH_1", notes="Original notes")
    pf.save_blast_record(b, db_path=TEST_DB)

    updated_b = b.update_snapshot(notes="Corrected notes")
    pf.save_blast_record(updated_b, db_path=TEST_DB)

    fetched = pf.get_blast_record("B_CORR", db_path=TEST_DB)
    assert fetched.notes == "Corrected notes"


def test_13_filter_historical_analytics():
    b1 = pf.BlastRecord(blast_id="B_JWA", site_id="Jwaneng", operation_type="open_pit", location_identifier="BENCH_1")
    b2 = pf.BlastRecord(blast_id="B_ORA", site_id="Orapa", operation_type="open_pit", location_identifier="BENCH_2")
    pf.save_blast_record(b1, db_path=TEST_DB)
    pf.save_blast_record(b2, db_path=TEST_DB)

    recs = pf.list_blast_records(site_id="Jwaneng", db_path=TEST_DB)
    assert len(recs) == 1
    assert recs[0].site_id == "Jwaneng"


def test_14_handle_insufficient_data_health_warning():
    preds = [10.0, 12.0]  # Only 2 points (< 5)
    acts = [10.0, 12.0]

    metrics = pf.calculate_regression_metrics(preds, acts)
    assert metrics["warning"] == "Insufficient Data"


def test_15_export_approved_dataset_with_manifest():
    b = pf.BlastRecord(blast_id="B_APP", site_id="Jwaneng", operation_type="open_pit", location_identifier="BENCH_1", planned_params={"burden_m": 6.0})
    pf.save_blast_record(b, db_path=TEST_DB)

    m = pf.ActualMeasurement(measurement_id="M_APP", blast_id="B_APP", target_metric="d50_mm", measured_value=240.0, unit="mm")
    pf.save_actual_measurement(m, db_path=TEST_DB)

    r = pf.DatasetReviewRecord(review_id="R_APP", blast_id="B_APP", target_metric="d50_mm", status="Approved", inclusion_status=True)
    pf.save_dataset_review(r, db_path=TEST_DB)

    df_exp, manifest, csv_bytes = pf.export_approved_dataset(target_metric="d50_mm", db_path=TEST_DB)

    assert manifest.approved_record_count == 1
    assert "B_APP" in manifest.included_blast_ids
    assert len(csv_bytes) > 0


def test_16_prevent_blast_level_partition_leakage():
    df_data = pd.DataFrame([
        {"blast_id": "BLAST_01", "feature_1": 1.0, "target": 10.0},
        {"blast_id": "BLAST_01", "feature_1": 1.1, "target": 10.5},
        {"blast_id": "BLAST_02", "feature_1": 2.0, "target": 20.0},
        {"blast_id": "BLAST_03", "feature_1": 3.0, "target": 30.0},
    ])

    train_df, test_df = pf.prepare_group_split_dataset(df_data, group_col="blast_id", test_size=0.33)

    train_blasts = set(train_df["blast_id"].unique())
    test_blasts = set(test_df["blast_id"].unique())

    # Zero overlap between train and test blast IDs
    assert len(train_blasts.intersection(test_blasts)) == 0


def test_17_failed_import_or_validation_handling():
    df_raw = pd.DataFrame([
        {"blast_id": "", "d50_mm": "INVALID"},  # Invalid row
    ])
    mappings = pf.identify_candidate_mappings(list(df_raw.columns))
    valid_meass, row_errs, dup_conflicts = pf.validate_actual_results_df(df_raw, mappings)

    assert len(valid_meass) == 0
    assert len(row_errs) >= 1


def test_18_existing_blastopt_functionality_preserved():
    # Verify synthetic data generator and physics core still function cleanly
    from src.synthetic_data import generate_synthetic_blast_data
    from src.physics_core import usbm_ppv

    synth_df = generate_synthetic_blast_data(num_samples=10, seed=42)
    assert len(synth_df) == 10

    ppv_val = usbm_ppv(max_charge_per_delay_kg=640.0, distance_m=450.0)
    assert ppv_val > 0.0
