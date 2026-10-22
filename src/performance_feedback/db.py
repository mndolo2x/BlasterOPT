"""
SQLite Persistence Layer for Blast Performance Feedback.

Manages durable storage in data/processed/blast_performance.db for blast records,
original prediction snapshots, actual field measurements, comparison results,
dataset review records, and export manifests.
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from src.performance_feedback.models import (
    BlastRecord,
    PredictionSnapshot,
    ActualMeasurement,
    ComparisonResult,
    DatasetReviewRecord,
    ExportManifest,
)

logger = logging.getLogger(__name__)

DB_PATH = "data/processed/blast_performance.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns SQLite connection with row factory configured."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes performance feedback database tables and indices if not present."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Blast Records Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blast_records (
                blast_id TEXT PRIMARY KEY,
                site_id TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                location_identifier TEXT NOT NULL,
                blast_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                engineer_id TEXT,
                status TEXT NOT NULL,
                design_id TEXT,
                design_version INTEGER,
                planned_params_json TEXT,
                geological_context_json TEXT,
                notes TEXT
            )
        """)

        # 2. Prediction Snapshots Table (Immutable)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_snapshots (
                prediction_run_id TEXT PRIMARY KEY,
                blast_id TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                training_dataset_ref TEXT,
                target_metric TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                unit TEXT NOT NULL,
                input_features_snapshot_json TEXT,
                uncertainty_bounds_json TEXT,
                warnings_json TEXT,
                site_config_version TEXT,
                provenance TEXT,
                FOREIGN KEY (blast_id) REFERENCES blast_records(blast_id)
            )
        """)

        # 3. Actual Measurements Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS actual_measurements (
                measurement_id TEXT PRIMARY KEY,
                blast_id TEXT NOT NULL,
                measurement_timestamp TEXT NOT NULL,
                target_metric TEXT NOT NULL,
                measured_value REAL NOT NULL,
                unit TEXT NOT NULL,
                measurement_method TEXT,
                instrument_id TEXT,
                sensor_distance_m REAL,
                source_filename TEXT,
                raw_attributes_json TEXT,
                FOREIGN KEY (blast_id) REFERENCES blast_records(blast_id)
            )
        """)

        # 4. Comparison Results Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparison_results (
                comparison_id TEXT PRIMARY KEY,
                blast_id TEXT NOT NULL,
                prediction_run_id TEXT NOT NULL,
                measurement_id TEXT NOT NULL,
                target_metric TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                measured_value REAL NOT NULL,
                signed_error REAL NOT NULL,
                absolute_error REAL NOT NULL,
                percentage_error REAL,
                unit TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                compared_at TEXT NOT NULL,
                FOREIGN KEY (blast_id) REFERENCES blast_records(blast_id),
                FOREIGN KEY (prediction_run_id) REFERENCES prediction_snapshots(prediction_run_id),
                FOREIGN KEY (measurement_id) REFERENCES actual_measurements(measurement_id)
            )
        """)

        # 5. Dataset Review Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_reviews (
                review_id TEXT PRIMARY KEY,
                blast_id TEXT NOT NULL,
                target_metric TEXT NOT NULL,
                status TEXT NOT NULL,
                inclusion_status INTEGER NOT NULL,
                exclusion_reason TEXT,
                reviewer_id TEXT,
                review_timestamp TEXT,
                comments TEXT,
                FOREIGN KEY (blast_id) REFERENCES blast_records(blast_id)
            )
        """)

        # 6. Export Manifests Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS export_manifests (
                export_id TEXT PRIMARY KEY,
                export_timestamp TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                target_metric TEXT NOT NULL,
                included_blast_ids_json TEXT NOT NULL,
                approved_record_count INTEGER NOT NULL,
                feature_schema_json TEXT NOT NULL,
                unit_definitions_json TEXT NOT NULL,
                filters_applied_json TEXT NOT NULL,
                manifest_hash TEXT NOT NULL
            )
        """)

        # Indices for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blast_site ON blast_records(site_id, operation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_blast ON prediction_snapshots(blast_id, target_metric)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meas_blast ON actual_measurements(blast_id, target_metric)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comp_blast ON comparison_results(blast_id, target_metric)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_blast ON dataset_reviews(blast_id, target_metric)")

        conn.commit()


# Initialize schema on module import
init_db()


# --- CRUD FUNCTIONS: BLAST RECORDS ---

def save_blast_record(record: BlastRecord, db_path: str = DB_PATH) -> BlastRecord:
    """Inserts or updates a BlastRecord in SQLite."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO blast_records (
                blast_id, site_id, operation_type, location_identifier, blast_date,
                created_at, engineer_id, status, design_id, design_version,
                planned_params_json, geological_context_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(blast_id) DO UPDATE SET
                site_id=excluded.site_id,
                operation_type=excluded.operation_type,
                location_identifier=excluded.location_identifier,
                blast_date=excluded.blast_date,
                engineer_id=excluded.engineer_id,
                status=excluded.status,
                design_id=excluded.design_id,
                design_version=excluded.design_version,
                planned_params_json=excluded.planned_params_json,
                geological_context_json=excluded.geological_context_json,
                notes=excluded.notes
        """, (
            record.blast_id,
            record.site_id,
            record.operation_type,
            record.location_identifier,
            record.blast_date,
            record.created_at,
            record.engineer_id,
            record.status,
            record.design_id,
            record.design_version,
            json.dumps(record.planned_params),
            json.dumps(record.geological_context),
            record.notes,
        ))
        conn.commit()
    return record


def get_blast_record(blast_id: str, db_path: str = DB_PATH) -> Optional[BlastRecord]:
    """Retrieves BlastRecord by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM blast_records WHERE blast_id = ?", (blast_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return BlastRecord(
            blast_id=row["blast_id"],
            site_id=row["site_id"],
            operation_type=row["operation_type"],
            location_identifier=row["location_identifier"],
            blast_date=row["blast_date"],
            created_at=row["created_at"],
            engineer_id=row["engineer_id"],
            status=row["status"],
            design_id=row["design_id"],
            design_version=row["design_version"],
            planned_params=json.loads(row["planned_params_json"] or "{}"),
            geological_context=json.loads(row["geological_context_json"] or "{}"),
            notes=row["notes"],
        )


def list_blast_records(
    site_id: Optional[str] = None,
    operation_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 500,
    db_path: str = DB_PATH,
) -> List[BlastRecord]:
    """Lists blast records matching optional filters."""
    query = "SELECT * FROM blast_records WHERE 1=1"
    params: List[Any] = []

    if site_id and site_id != "ALL":
        query += " AND site_id = ?"
        params.append(site_id)
    if operation_type and operation_type != "ALL":
        query += " AND operation_type = ?"
        params.append(operation_type)
    if status and status != "ALL":
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY blast_date DESC LIMIT ?"
    params.append(limit)

    records = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            records.append(BlastRecord(
                blast_id=row["blast_id"],
                site_id=row["site_id"],
                operation_type=row["operation_type"],
                location_identifier=row["location_identifier"],
                blast_date=row["blast_date"],
                created_at=row["created_at"],
                engineer_id=row["engineer_id"],
                status=row["status"],
                design_id=row["design_id"],
                design_version=row["design_version"],
                planned_params=json.loads(row["planned_params_json"] or "{}"),
                geological_context=json.loads(row["geological_context_json"] or "{}"),
                notes=row["notes"],
            ))
    return records


# --- CRUD FUNCTIONS: PREDICTION SNAPSHOTS ---

def save_prediction_snapshot(snapshot: PredictionSnapshot, db_path: str = DB_PATH) -> PredictionSnapshot:
    """Saves an immutable PredictionSnapshot in SQLite."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO prediction_snapshots (
                prediction_run_id, blast_id, prediction_timestamp, model_name, model_version,
                training_dataset_ref, target_metric, predicted_value, unit,
                input_features_snapshot_json, uncertainty_bounds_json, warnings_json,
                site_config_version, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.prediction_run_id,
            snapshot.blast_id,
            snapshot.prediction_timestamp,
            snapshot.model_name,
            snapshot.model_version,
            snapshot.training_dataset_ref,
            snapshot.target_metric,
            snapshot.predicted_value,
            snapshot.unit,
            json.dumps(snapshot.input_features_snapshot),
            json.dumps(snapshot.uncertainty_bounds) if snapshot.uncertainty_bounds else None,
            json.dumps(snapshot.warnings),
            snapshot.site_config_version,
            snapshot.provenance,
        ))
        conn.commit()
    return snapshot


def get_prediction_snapshots_for_blast(
    blast_id: str, target_metric: Optional[str] = None, db_path: str = DB_PATH
) -> List[PredictionSnapshot]:
    """Retrieves all prediction snapshots for a blast ID."""
    query = "SELECT * FROM prediction_snapshots WHERE blast_id = ?"
    params: List[Any] = [blast_id]
    if target_metric:
        query += " AND target_metric = ?"
        params.append(target_metric)
    query += " ORDER BY prediction_timestamp DESC"

    snapshots = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            snapshots.append(PredictionSnapshot(
                prediction_run_id=row["prediction_run_id"],
                blast_id=row["blast_id"],
                prediction_timestamp=row["prediction_timestamp"],
                model_name=row["model_name"],
                model_version=row["model_version"],
                training_dataset_ref=row["training_dataset_ref"],
                target_metric=row["target_metric"],
                predicted_value=row["predicted_value"],
                unit=row["unit"],
                input_features_snapshot=json.loads(row["input_features_snapshot_json"] or "{}"),
                uncertainty_bounds=json.loads(row["uncertainty_bounds_json"]) if row["uncertainty_bounds_json"] else None,
                warnings=json.loads(row["warnings_json"] or "[]"),
                site_config_version=row["site_config_version"],
                provenance=row["provenance"],
            ))
    return snapshots


# --- CRUD FUNCTIONS: ACTUAL MEASUREMENTS ---

def save_actual_measurement(meas: ActualMeasurement, db_path: str = DB_PATH) -> ActualMeasurement:
    """Inserts or updates an ActualMeasurement in SQLite."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO actual_measurements (
                measurement_id, blast_id, measurement_timestamp, target_metric,
                measured_value, unit, measurement_method, instrument_id,
                sensor_distance_m, source_filename, raw_attributes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(measurement_id) DO UPDATE SET
                blast_id=excluded.blast_id,
                measurement_timestamp=excluded.measurement_timestamp,
                target_metric=excluded.target_metric,
                measured_value=excluded.measured_value,
                unit=excluded.unit,
                measurement_method=excluded.measurement_method,
                instrument_id=excluded.instrument_id,
                sensor_distance_m=excluded.sensor_distance_m,
                source_filename=excluded.source_filename,
                raw_attributes_json=excluded.raw_attributes_json
        """, (
            meas.measurement_id,
            meas.blast_id,
            meas.measurement_timestamp,
            meas.target_metric,
            float(meas.measured_value) if isinstance(meas.measured_value, (int, float)) else str(meas.measured_value),
            meas.unit,
            meas.measurement_method,
            meas.instrument_id,
            meas.sensor_distance_m,
            meas.source_filename,
            json.dumps(meas.raw_attributes),
        ))
        conn.commit()
    return meas


def get_actual_measurements_for_blast(
    blast_id: str, target_metric: Optional[str] = None, db_path: str = DB_PATH
) -> List[ActualMeasurement]:
    """Retrieves actual measurements for a blast ID."""
    query = "SELECT * FROM actual_measurements WHERE blast_id = ?"
    params: List[Any] = [blast_id]
    if target_metric:
        query += " AND target_metric = ?"
        params.append(target_metric)
    query += " ORDER BY measurement_timestamp DESC"

    measurements = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            measurements.append(ActualMeasurement(
                measurement_id=row["measurement_id"],
                blast_id=row["blast_id"],
                measurement_timestamp=row["measurement_timestamp"],
                target_metric=row["target_metric"],
                measured_value=row["measured_value"],
                unit=row["unit"],
                measurement_method=row["measurement_method"],
                instrument_id=row["instrument_id"],
                sensor_distance_m=row["sensor_distance_m"],
                source_filename=row["source_filename"],
                raw_attributes=json.loads(row["raw_attributes_json"] or "{}"),
            ))
    return measurements


# --- CRUD FUNCTIONS: COMPARISON RESULTS ---

def save_comparison_result(comp: ComparisonResult, db_path: str = DB_PATH) -> ComparisonResult:
    """Inserts or updates a ComparisonResult."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO comparison_results (
                comparison_id, blast_id, prediction_run_id, measurement_id,
                target_metric, predicted_value, measured_value, signed_error,
                absolute_error, percentage_error, unit, model_name, model_version, compared_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(comparison_id) DO UPDATE SET
                predicted_value=excluded.predicted_value,
                measured_value=excluded.measured_value,
                signed_error=excluded.signed_error,
                absolute_error=excluded.absolute_error,
                percentage_error=excluded.percentage_error,
                compared_at=excluded.compared_at
        """, (
            comp.comparison_id,
            comp.blast_id,
            comp.prediction_run_id,
            comp.measurement_id,
            comp.target_metric,
            comp.predicted_value,
            comp.measured_value,
            comp.signed_error,
            comp.absolute_error,
            comp.percentage_error,
            comp.unit,
            comp.model_name,
            comp.model_version,
            comp.compared_at,
        ))
        conn.commit()
    return comp


def list_comparison_results(
    site_id: Optional[str] = None,
    target_metric: Optional[str] = None,
    model_name: Optional[str] = None,
    operation_type: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[ComparisonResult]:
    """Lists comparison results matching filters."""
    query = """
        SELECT c.* FROM comparison_results c
        JOIN blast_records b ON c.blast_id = b.blast_id
        WHERE 1=1
    """
    params: List[Any] = []

    if site_id and site_id != "ALL":
        query += " AND b.site_id = ?"
        params.append(site_id)
    if target_metric and target_metric != "ALL":
        query += " AND c.target_metric = ?"
        params.append(target_metric)
    if model_name and model_name != "ALL":
        query += " AND c.model_name = ?"
        params.append(model_name)
    if operation_type and operation_type != "ALL":
        query += " AND b.operation_type = ?"
        params.append(operation_type)

    query += " ORDER BY c.compared_at DESC"

    results = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            results.append(ComparisonResult(
                comparison_id=row["comparison_id"],
                blast_id=row["blast_id"],
                prediction_run_id=row["prediction_run_id"],
                measurement_id=row["measurement_id"],
                target_metric=row["target_metric"],
                predicted_value=row["predicted_value"],
                measured_value=row["measured_value"],
                signed_error=row["signed_error"],
                absolute_error=row["absolute_error"],
                percentage_error=row["percentage_error"],
                unit=row["unit"],
                model_name=row["model_name"],
                model_version=row["model_version"],
                compared_at=row["compared_at"],
            ))
    return results


# --- CRUD FUNCTIONS: DATASET REVIEWS ---

def save_dataset_review(review: DatasetReviewRecord, db_path: str = DB_PATH) -> DatasetReviewRecord:
    """Inserts or updates a DatasetReviewRecord."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dataset_reviews (
                review_id, blast_id, target_metric, status,
                inclusion_status, exclusion_reason, reviewer_id,
                review_timestamp, comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(review_id) DO UPDATE SET
                status=excluded.status,
                inclusion_status=excluded.inclusion_status,
                exclusion_reason=excluded.exclusion_reason,
                reviewer_id=excluded.reviewer_id,
                review_timestamp=excluded.review_timestamp,
                comments=excluded.comments
        """, (
            review.review_id,
            review.blast_id,
            review.target_metric,
            review.status,
            1 if review.inclusion_status else 0,
            review.exclusion_reason,
            review.reviewer_id,
            review.review_timestamp,
            review.comments,
        ))
        conn.commit()
    return review


def list_dataset_reviews(
    target_metric: Optional[str] = None, status: Optional[str] = None, db_path: str = DB_PATH
) -> List[DatasetReviewRecord]:
    """Lists dataset review records matching filters."""
    query = "SELECT * FROM dataset_reviews WHERE 1=1"
    params: List[Any] = []
    if target_metric and target_metric != "ALL":
        query += " AND target_metric = ?"
        params.append(target_metric)
    if status and status != "ALL":
        query += " AND status = ?"
        params.append(status)

    reviews = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            reviews.append(DatasetReviewRecord(
                review_id=row["review_id"],
                blast_id=row["blast_id"],
                target_metric=row["target_metric"],
                status=row["status"],
                inclusion_status=bool(row["inclusion_status"]),
                exclusion_reason=row["exclusion_reason"],
                reviewer_id=row["reviewer_id"],
                review_timestamp=row["review_timestamp"],
                comments=row["comments"],
            ))
    return reviews


# --- CRUD FUNCTIONS: EXPORT MANIFESTS ---

def save_export_manifest(manifest: ExportManifest, db_path: str = DB_PATH) -> ExportManifest:
    """Saves an ExportManifest record."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO export_manifests (
                export_id, export_timestamp, dataset_version, target_metric,
                included_blast_ids_json, approved_record_count, feature_schema_json,
                unit_definitions_json, filters_applied_json, manifest_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            manifest.export_id,
            manifest.export_timestamp,
            manifest.dataset_version,
            manifest.target_metric,
            json.dumps(manifest.included_blast_ids),
            manifest.approved_record_count,
            json.dumps(manifest.feature_schema),
            json.dumps(manifest.unit_definitions),
            json.dumps(manifest.filters_applied),
            manifest.manifest_hash,
        ))
        conn.commit()
    return manifest
