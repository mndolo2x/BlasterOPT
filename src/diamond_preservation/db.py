"""
SQLite Persistence Layer for Diamond Preservation.

Manages durable storage in data/processed/diamond_preservation.db for observations,
breakage assessments, scenarios, valuation schedules, and audit records.
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from src.diamond_preservation.models import (
    DiamondObservation,
    DiamondAssessment,
    DiamondScenario,
    DiamondValuationSchedule,
)

logger = logging.getLogger(__name__)

DB_PATH = "data/processed/diamond_preservation.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns SQLite connection with row factory configured."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes diamond preservation database tables if not present."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Observations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diamond_observations (
                observation_id TEXT PRIMARY KEY,
                blast_id TEXT NOT NULL,
                sample_lot_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                observation_method TEXT NOT NULL,
                stone_size_class TEXT NOT NULL,
                stone_count INTEGER NOT NULL,
                total_carats REAL NOT NULL,
                stone_quality_category TEXT NOT NULL,
                damage_type TEXT NOT NULL,
                damage_attribution_cause TEXT NOT NULL,
                attribution_confidence REAL NOT NULL,
                observer_id TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                notes TEXT
            )
        """)

        # 2. Assessments Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diamond_assessments (
                assessment_id TEXT PRIMARY KEY,
                blast_id TEXT NOT NULL,
                assessment_timestamp TEXT NOT NULL,
                assessor_id TEXT NOT NULL,
                completeness_score_pct REAL NOT NULL,
                missing_mandatory_fields_json TEXT,
                missing_optional_fields_json TEXT,
                specific_charge_kg_t REAL,
                max_charge_per_delay_kg REAL,
                vibration_ppv_mms REAL,
                model_prediction_status TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                input_snapshot_json TEXT,
                provenance TEXT NOT NULL
            )
        """)

        # 3. Scenarios Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diamond_scenarios (
                scenario_id TEXT PRIMARY KEY,
                scenario_name TEXT NOT NULL,
                baseline_blast_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                candidate_params_json TEXT NOT NULL,
                parameter_deltas_json TEXT NOT NULL,
                feasibility_status TEXT NOT NULL,
                warning_messages_json TEXT
            )
        """)

        # 4. Valuation Schedules Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valuation_schedules (
                schedule_id TEXT PRIMARY KEY,
                schedule_name TEXT NOT NULL,
                effective_date TEXT NOT NULL,
                currency TEXT NOT NULL,
                valuation_map_json TEXT NOT NULL,
                source TEXT NOT NULL
            )
        """)

        # Indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_blast ON diamond_observations(blast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ass_blast ON diamond_assessments(blast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scen_base ON diamond_scenarios(baseline_blast_id)")

        conn.commit()


init_db()


# --- CRUD FUNCTIONS ---

def save_diamond_observation(obs: DiamondObservation, db_path: str = DB_PATH) -> DiamondObservation:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO diamond_observations (
                observation_id, blast_id, sample_lot_id, stage, observation_method,
                stone_size_class, stone_count, total_carats, stone_quality_category,
                damage_type, damage_attribution_cause, attribution_confidence,
                observer_id, observation_date, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(observation_id) DO UPDATE SET
                stage=excluded.stage,
                stone_size_class=excluded.stone_size_class,
                stone_count=excluded.stone_count,
                total_carats=excluded.total_carats,
                damage_type=excluded.damage_type,
                damage_attribution_cause=excluded.damage_attribution_cause,
                attribution_confidence=excluded.attribution_confidence,
                notes=excluded.notes
        """, (
            obs.observation_id, obs.blast_id, obs.sample_lot_id, obs.stage, obs.observation_method,
            obs.stone_size_class, obs.stone_count, obs.total_carats, obs.stone_quality_category,
            obs.damage_type, obs.damage_attribution_cause, obs.attribution_confidence,
            obs.observer_id, obs.observation_date, obs.notes
        ))
        conn.commit()
    return obs


def list_diamond_observations(blast_id: Optional[str] = None, stage: Optional[str] = None, db_path: str = DB_PATH) -> List[DiamondObservation]:
    query = "SELECT * FROM diamond_observations WHERE 1=1"
    params: List[Any] = []
    if blast_id and blast_id != "ALL":
        query += " AND blast_id = ?"
        params.append(blast_id)
    if stage and stage != "ALL":
        query += " AND stage = ?"
        params.append(stage)

    query += " ORDER BY observation_date DESC"

    res = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            res.append(DiamondObservation(
                observation_id=row["observation_id"],
                blast_id=row["blast_id"],
                sample_lot_id=row["sample_lot_id"],
                stage=row["stage"],
                observation_method=row["observation_method"],
                stone_size_class=row["stone_size_class"],
                stone_count=row["stone_count"],
                total_carats=row["total_carats"],
                stone_quality_category=row["stone_quality_category"],
                damage_type=row["damage_type"],
                damage_attribution_cause=row["damage_attribution_cause"],
                attribution_confidence=row["attribution_confidence"],
                observer_id=row["observer_id"],
                observation_date=row["observation_date"],
                notes=row["notes"],
            ))
    return res


def save_diamond_assessment(ass: DiamondAssessment, db_path: str = DB_PATH) -> DiamondAssessment:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO diamond_assessments (
                assessment_id, blast_id, assessment_timestamp, assessor_id,
                completeness_score_pct, missing_mandatory_fields_json,
                missing_optional_fields_json, specific_charge_kg_t,
                max_charge_per_delay_kg, vibration_ppv_mms,
                model_prediction_status, risk_category, input_snapshot_json, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(assessment_id) DO UPDATE SET
                completeness_score_pct=excluded.completeness_score_pct,
                specific_charge_kg_t=excluded.specific_charge_kg_t,
                risk_category=excluded.risk_category,
                input_snapshot_json=excluded.input_snapshot_json
        """, (
            ass.assessment_id, ass.blast_id, ass.assessment_timestamp, ass.assessor_id,
            ass.completeness_score_pct, json.dumps(ass.missing_mandatory_fields),
            json.dumps(ass.missing_optional_fields), ass.specific_charge_kg_t,
            ass.max_charge_per_delay_kg, ass.vibration_ppv_mms,
            ass.model_prediction_status, ass.risk_category,
            json.dumps(ass.input_snapshot), ass.provenance
        ))
        conn.commit()
    return ass


def list_diamond_assessments(blast_id: Optional[str] = None, db_path: str = DB_PATH) -> List[DiamondAssessment]:
    query = "SELECT * FROM diamond_assessments WHERE 1=1"
    params: List[Any] = []
    if blast_id and blast_id != "ALL":
        query += " AND blast_id = ?"
        params.append(blast_id)

    query += " ORDER BY assessment_timestamp DESC"

    res = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            res.append(DiamondAssessment(
                assessment_id=row["assessment_id"],
                blast_id=row["blast_id"],
                assessment_timestamp=row["assessment_timestamp"],
                assessor_id=row["assessor_id"],
                completeness_score_pct=row["completeness_score_pct"],
                missing_mandatory_fields=json.loads(row["missing_mandatory_fields_json"] or "[]"),
                missing_optional_fields=json.loads(row["missing_optional_fields_json"] or "[]"),
                specific_charge_kg_t=row["specific_charge_kg_t"],
                max_charge_per_delay_kg=row["max_charge_per_delay_kg"],
                vibration_ppv_mms=row["vibration_ppv_mms"],
                model_prediction_status=row["model_prediction_status"],
                risk_category=row["risk_category"],
                input_snapshot=json.loads(row["input_snapshot_json"] or "{}"),
                provenance=row["provenance"],
            ))
    return res


def save_diamond_scenario(scen: DiamondScenario, db_path: str = DB_PATH) -> DiamondScenario:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO diamond_scenarios (
                scenario_id, scenario_name, baseline_blast_id, created_at,
                created_by, candidate_params_json, parameter_deltas_json,
                feasibility_status, warning_messages_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scenario_id) DO UPDATE SET
                scenario_name=excluded.scenario_name,
                candidate_params_json=excluded.candidate_params_json,
                parameter_deltas_json=excluded.parameter_deltas_json,
                feasibility_status=excluded.feasibility_status,
                warning_messages_json=excluded.warning_messages_json
        """, (
            scen.scenario_id, scen.scenario_name, scen.baseline_blast_id,
            scen.created_at, scen.created_by, json.dumps(scen.candidate_params),
            json.dumps(scen.parameter_deltas), scen.feasibility_status,
            json.dumps(scen.warning_messages)
        ))
        conn.commit()
    return scen


def list_diamond_scenarios(baseline_blast_id: Optional[str] = None, db_path: str = DB_PATH) -> List[DiamondScenario]:
    query = "SELECT * FROM diamond_scenarios WHERE 1=1"
    params: List[Any] = []
    if baseline_blast_id and baseline_blast_id != "ALL":
        query += " AND baseline_blast_id = ?"
        params.append(baseline_blast_id)

    query += " ORDER BY created_at DESC"

    res = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            res.append(DiamondScenario(
                scenario_id=row["scenario_id"],
                scenario_name=row["scenario_name"],
                baseline_blast_id=row["baseline_blast_id"],
                created_at=row["created_at"],
                created_by=row["created_by"],
                candidate_params=json.loads(row["candidate_params_json"] or "{}"),
                parameter_deltas=json.loads(row["parameter_deltas_json"] or "{}"),
                feasibility_status=row["feasibility_status"],
                warning_messages=json.loads(row["warning_messages_json"] or "[]"),
            ))
    return res


def save_valuation_schedule(sched: DiamondValuationSchedule, db_path: str = DB_PATH) -> DiamondValuationSchedule:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO valuation_schedules (
                schedule_id, schedule_name, effective_date, currency, valuation_map_json, source
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(schedule_id) DO UPDATE SET
                schedule_name=excluded.schedule_name,
                effective_date=excluded.effective_date,
                valuation_map_json=excluded.valuation_map_json
        """, (
            sched.schedule_id, sched.schedule_name, sched.effective_date,
            sched.currency, json.dumps(sched.valuation_map), sched.source
        ))
        conn.commit()
    return sched


def get_latest_valuation_schedule(db_path: str = DB_PATH) -> DiamondValuationSchedule:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM valuation_schedules ORDER BY effective_date DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            # Save default
            default_sched = DiamondValuationSchedule(
                schedule_id="SCHED_DEBSWANA_Q4_2026",
                schedule_name="Q4 2026 Debswana Price Schedule",
            )
            return save_valuation_schedule(default_sched, db_path)

        return DiamondValuationSchedule(
            schedule_id=row["schedule_id"],
            schedule_name=row["schedule_name"],
            effective_date=row["effective_date"],
            currency=row["currency"],
            valuation_map=json.loads(row["valuation_map_json"] or "{}"),
            source=row["source"],
        )
