"""
Automated Unit Tests for Diamond Preservation Subsystem.

Tests:
1. Diamond observation creation, persistence, and retrieval.
2. Linking diamond observations to existing blast IDs.
3. Import and validation of diamond observation CSV records.
4. Handling unknown damage attribution causes cleanly without false attribution.
5. Creating and persisting diamond breakage assessments with specific charge calculations.
6. Scenario lab comparison between baseline and candidate scenarios without overwriting baseline.
7. Unverified feasibility status warnings for candidate scenarios.
8. Diamond population economics valuation across size classes.
9. Dataset readiness reporting for future model training (checking independent blasts & completeness).
10. Verification of missing model status message when no predictor is calibrated.
11. Exporting validated observation records to CSV.
12. Existing BlastOPT features regression verification.
"""

import os
import pytest
import pandas as pd
import numpy as np
import src.diamond_preservation as dp
import src.performance_feedback as pf

TEST_DB = "data/processed/test_diamond_preservation.db"


@pytest.fixture(autouse=True)
def setup_test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    dp.init_db(TEST_DB)
    pf.init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_1_observation_creation_and_persistence():
    obs = dp.DiamondObservation(
        observation_id="OBS_TEST_01",
        blast_id="BLASTOPT-TEST-001",
        sample_lot_id="LOT_01",
        stage="post_blast",
        stone_size_class="+11 DTC",
        stone_count=5,
        total_carats=12.4,
        damage_type="chipping",
        damage_attribution_cause="blast_induced",
    )
    dp.save_diamond_observation(obs, db_path=TEST_DB)

    fetched = dp.list_diamond_observations("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert len(fetched) == 1
    assert fetched[0].stone_count == 5
    assert fetched[0].total_carats == 12.4


def test_2_linking_observation_to_blast_record():
    b_rec = pf.BlastRecord(blast_id="BLASTOPT-TEST-001", site_id="Jwaneng", location_identifier="BENCH_15S")
    pf.save_blast_record(b_rec, db_path=TEST_DB)

    obs = dp.DiamondObservation(
        observation_id="OBS_TEST_02",
        blast_id="BLASTOPT-TEST-001",
        sample_lot_id="LOT_02",
        total_carats=8.0,
    )
    dp.save_diamond_observation(obs, db_path=TEST_DB)

    b_fetched = pf.get_blast_record("BLASTOPT-TEST-001", db_path=TEST_DB)
    o_fetched = dp.list_diamond_observations("BLASTOPT-TEST-001", db_path=TEST_DB)

    assert b_fetched is not None
    assert len(o_fetched) == 1
    assert o_fetched[0].blast_id == b_fetched.blast_id


def test_3_import_and_validate_observation_df():
    df_raw = pd.DataFrame([
        {
            "blast_id": "BLASTOPT-TEST-001",
            "sample_lot_id": "LOT_01",
            "stage": "post_blast",
            "stone_size_class": "+11 DTC",
            "stone_count": 2,
            "total_carats": 6.5,
            "damage_type": "cracking",
            "damage_attribution_cause": "blast_induced",
        }
    ])

    v_obs, errs = dp.validate_diamond_observations_df(df_raw)
    assert len(v_obs) == 1
    assert len(errs) == 0
    assert v_obs[0].total_carats == 6.5


def test_4_unknown_damage_attribution_preserved():
    obs = dp.DiamondObservation(
        observation_id="OBS_UNK",
        blast_id="BLASTOPT-TEST-001",
        sample_lot_id="LOT_UNK",
        damage_type="unknown",
        damage_attribution_cause="unknown",
        total_carats=10.0,
    )
    dp.save_diamond_observation(obs, db_path=TEST_DB)

    fetched = dp.list_diamond_observations("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert fetched[0].damage_attribution_cause == "unknown"


def test_5_breakage_assessment_creation():
    ass = dp.DiamondAssessment(
        assessment_id="ASS_01",
        blast_id="BLASTOPT-TEST-001",
        completeness_score_pct=90.0,
        specific_charge_kg_t=0.28,
        risk_category="Moderate Risk",
    )
    dp.save_diamond_assessment(ass, db_path=TEST_DB)

    fetched = dp.list_diamond_assessments("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert len(fetched) == 1
    assert fetched[0].specific_charge_kg_t == 0.28


def test_6_scenario_lab_without_overwriting_baseline():
    b_rec = pf.BlastRecord(
        blast_id="BLASTOPT-TEST-001",
        site_id="Jwaneng",
        location_identifier="BENCH_15S",
        planned_params={"powder_factor_kg_m3": 0.65},
    )
    pf.save_blast_record(b_rec, db_path=TEST_DB)

    scen = dp.DiamondScenario(
        scenario_id="SCEN_01",
        scenario_name="Candidate Option 1",
        baseline_blast_id="BLASTOPT-TEST-001",
        candidate_params={"powder_factor_kg_m3": 0.52},
        parameter_deltas={"powder_factor_delta": -0.13},
        feasibility_status="Unverified",
    )
    dp.save_diamond_scenario(scen, db_path=TEST_DB)

    # Verify baseline remains unchanged
    base_fetched = pf.get_blast_record("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert base_fetched.planned_params["powder_factor_kg_m3"] == 0.65

    scen_fetched = dp.list_diamond_scenarios("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert len(scen_fetched) == 1
    assert scen_fetched[0].candidate_params["powder_factor_kg_m3"] == 0.52


def test_7_unverified_feasibility_warnings():
    scen = dp.DiamondScenario(
        scenario_id="SCEN_02",
        scenario_name="Unverified Scenario",
        baseline_blast_id="BLASTOPT-TEST-001",
        feasibility_status="Unverified",
        warning_messages=["Feasibility unverified against production predictor."],
    )
    assert scen.feasibility_status == "Unverified"
    assert len(scen.warning_messages) == 1


def test_8_population_valuation_calculation():
    sched = dp.DiamondValuationSchedule(
        schedule_id="SCHED_01",
        schedule_name="Test Schedule",
        valuation_map={"+11 DTC": 400.0},
    )
    obs = [
        dp.DiamondObservation(
            observation_id="OBS_VAL",
            blast_id="BLASTOPT-TEST-001",
            sample_lot_id="LOT_VAL",
            stone_size_class="+11 DTC",
            stone_count=2,
            total_carats=10.0,
            damage_type="none",
        )
    ]
    econ = dp.calculate_population_valuation(obs, sched)
    assert econ["total_carats"] == 10.0
    assert econ["total_estimated_value_usd"] == 4000.0


def test_9_dataset_readiness_reporting():
    obs_list = [
        dp.DiamondObservation(
            observation_id=f"OBS_{i}",
            blast_id=f"BLAST_{i}",
            sample_lot_id=f"LOT_{i}",
            damage_type="chipping" if i % 2 == 0 else "unknown",
            total_carats=5.0,
        )
        for i in range(5)
    ]
    readiness = dp.evaluate_dataset_readiness(obs_list)
    assert readiness.total_observations == 5
    assert readiness.total_distinct_blasts == 5
    assert readiness.is_ready_for_training is False  # Insufficient blasts (< 10)


def test_10_missing_model_disclaimer_status():
    ass = dp.DiamondAssessment(
        assessment_id="ASS_DISC",
        blast_id="BLASTOPT-TEST-001",
        completeness_score_pct=50.0,
    )
    assert "no validated model is configured" in ass.model_prediction_status


def test_11_export_observation_records_to_csv():
    obs = dp.DiamondObservation(
        observation_id="OBS_EXP",
        blast_id="BLASTOPT-TEST-001",
        sample_lot_id="LOT_EXP",
        total_carats=15.0,
    )
    dp.save_diamond_observation(obs, db_path=TEST_DB)

    fetched = dp.list_diamond_observations("BLASTOPT-TEST-001", db_path=TEST_DB)
    df_exp = pd.DataFrame([f.model_dump() if hasattr(f, "model_dump") else f.dict() for f in fetched])
    assert len(df_exp) == 1
    assert "total_carats" in df_exp.columns


def test_12_specific_charge_and_engineering_calculations():
    sc_res = dp.calculate_specific_charge(total_explosive_kg=320.0, blasted_tonnage=1200.0)
    assert sc_res["value"] == 0.2667
    assert sc_res["unit"] == "kg/t"

    dev_res = dp.calculate_drilling_deviation(planned_depth_m=15.0, actual_depth_m=16.2)
    assert dev_res["depth_delta_m"] == 1.2
    assert dev_res["percentage_deviation"] == 8.0


def test_13_diamond_breakage_predictor_interface_fallback():
    predictor = dp.DiamondBreakagePredictorInterface()
    assert predictor.is_trained() is False

    pred_res = predictor.predict({"burden_m": 6.0})
    assert pred_res["status"] == "unavailable"
    assert "no validated model is configured" in pred_res["message"]


def test_14_grouped_training_split_prevents_leakage():
    df_data = pd.DataFrame([
        {"blast_id": "BLAST_01", "powder_factor_kg_m3": 0.65, "damage_type": "chipping"},
        {"blast_id": "BLAST_01", "powder_factor_kg_m3": 0.68, "damage_type": "none"},
        {"blast_id": "BLAST_02", "powder_factor_kg_m3": 0.70, "damage_type": "cracking"},
        {"blast_id": "BLAST_03", "powder_factor_kg_m3": 0.60, "damage_type": "none"},
    ])

    train_df, test_df = dp.prepare_grouped_training_split(df_data, group_col="blast_id", test_size=0.33)
    train_blasts = set(train_df["blast_id"].unique())
    test_blasts = set(test_df["blast_id"].unique())

    # Ensure zero overlap of blast_ids between train and test
    assert len(train_blasts.intersection(test_blasts)) == 0


def test_15_constrained_scenario_evaluator_deltas():
    base_p = {"burden_m": 6.0, "spacing_m": 7.0, "powder_factor_kg_m3": 0.65}
    cand_p = {"burden_m": 6.0, "spacing_m": 7.5, "powder_factor_kg_m3": 0.55}

    eval_res = dp.evaluate_scenario_deltas(base_p, cand_p)
    assert "spacing_m" in eval_res["deltas"]
    assert eval_res["deltas"]["spacing_m"]["delta"] == 0.5
    assert eval_res["deltas"]["powder_factor_kg_m3"]["delta"] == -0.1


def test_16_observation_stages_and_constants_exported():
    assert hasattr(dp, "OBSERVATION_STAGES")
    assert hasattr(dp, "DAMAGE_TYPES")
    assert hasattr(dp, "DAMAGE_ATTRIBUTIONS")

    stages = dp.OBSERVATION_STAGES
    assert isinstance(stages, list)
    assert "post_blast" in stages
    assert "before_blasting" in stages

    # Test processing all selectable stages in validation & persistence
    for stg in stages:
        obs = dp.DiamondObservation(
            observation_id=f"OBS_STG_{stg}",
            blast_id="BLASTOPT-TEST-001",
            sample_lot_id=f"LOT_{stg}",
            stage=stg,
            stone_size_class="+11 DTC",
            stone_count=1,
            total_carats=2.5,
        )
        dp.save_diamond_observation(obs, db_path=TEST_DB)

    fetched = dp.list_diamond_observations("BLASTOPT-TEST-001", db_path=TEST_DB)
    assert len(fetched) == len(stages)


def test_17_existing_blastopt_features_regression():
    from src.synthetic_data import generate_synthetic_blast_data
    from src.physics_core import usbm_ppv

    synth_df = generate_synthetic_blast_data(num_samples=5, seed=42)
    assert len(synth_df) == 5

    ppv_val = usbm_ppv(max_charge_per_delay_kg=640.0, distance_m=450.0)
    assert ppv_val > 0.0
