"""Validate and append historical records to the existing BlasterOPT databases."""

import csv
import math
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from src.diamond_preservation import db as diamond_db
from src.diamond_preservation.ingestion import DAMAGE_TYPES, OBSERVATION_STAGES
from src.performance_feedback import db as performance_db
from src.performance_feedback.ingestion import TARGET_UNITS

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 1000
MAX_COLUMNS = 64

MEASUREMENT_COLUMNS = (
    "measurement_id", "blast_id", "site_id", "operation_type", "location_identifier",
    "blast_date", "measurement_timestamp", "target_metric", "measured_value", "unit",
)
DIAMOND_COLUMNS = (
    "observation_id", "blast_id", "sample_lot_id", "observation_date", "stage",
    "stone_size_class", "stone_count", "total_carats", "damage_type", "observer_id",
)
COLUMNS = {"measurements": MEASUREMENT_COLUMNS, "diamonds": DIAMOND_COLUMNS}


def read_sheet(filename: str, data: bytes):
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        yield from csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        return
    if suffix != ".xlsx":
        raise ValueError("Choose a CSV or .xlsx file.")

    # Excel workbooks are ZIP archives. Check expanded size before XML parsing.
    with ZipFile(BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 256 or sum(member.file_size for member in members) > 24 * MAX_BYTES:
            raise ValueError("The expanded Excel workbook is too large.")
    workbook = load_workbook(BytesIO(data), read_only=True, data_only=False, keep_links=False)
    try:
        if not workbook.worksheets:
            raise ValueError("The workbook has no worksheets.")
        sheet = workbook.worksheets[0]
        if sheet.max_column and sheet.max_column > MAX_COLUMNS:
            raise ValueError(f"Use at most {MAX_COLUMNS} columns.")
        # Some spreadsheet tools write inaccurate dimensions. Read the actual rows.
        sheet.reset_dimensions()
        for cells in sheet.iter_rows():
            yield [cell.value for cell in cells]
    finally:
        workbook.close()


def iso_date(value: str, column: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{column} must be an ISO date or timestamp, such as 2026-11-12.") from None
    if len(value) == 10:
        return parsed.date().isoformat()
    return (parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed).isoformat()


def validate_row(kind: str, row: dict) -> dict:
    missing = [name for name, value in row.items() if not value]
    if missing:
        raise ValueError("Missing values for " + ", ".join(missing) + ".")
    if any(len(value) > 200 or "\x00" in value or value.startswith("=") for value in row.values()):
        raise ValueError("Use plain values up to 200 characters. Replace formulas with their values.")

    if kind == "measurements":
        if row["operation_type"] not in ("open_pit", "underground"):
            raise ValueError("operation_type must be open_pit or underground.")
        row["blast_date"] = iso_date(row["blast_date"], "blast_date")
        row["measurement_timestamp"] = iso_date(row["measurement_timestamp"], "measurement_timestamp")
        metric, unit = row["target_metric"], row["unit"]
        if metric in TARGET_UNITS and unit != TARGET_UNITS[metric]:
            raise ValueError(f"Unit for {metric} must be {TARGET_UNITS[metric]}.")
        row["measured_value"] = finite_number(row["measured_value"], "measured_value")
    else:
        row["observation_date"] = iso_date(row["observation_date"], "observation_date")
        if row["stage"] not in OBSERVATION_STAGES or row["damage_type"] not in DAMAGE_TYPES:
            raise ValueError("Use a listed stage and damage_type from the template instructions.")
        count = finite_number(row["stone_count"], "stone_count")
        carats = finite_number(row["total_carats"], "total_carats")
        if not count.is_integer() or not 0 < count <= 2**31 - 1 or carats <= 0:
            raise ValueError("stone_count must be a positive whole number and total_carats must be positive.")
        row["stone_count"] = int(count)
        row["total_carats"] = carats
    return row


def finite_number(value: str, column: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{column} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{column} must be finite.")
    return number


def preview_import(kind: str, filename: str, data: bytes) -> tuple[list[dict], list[str]]:
    required = COLUMNS[kind]
    records, errors = [], []
    identifiers, blasts = set(), {}
    if len(data) > MAX_BYTES:
        return records, ["Choose a file up to 2 MiB."]
    try:
        with closing(read_sheet(filename, data)) as rows:
            headers = [str(value or "").strip().lower() for value in next(rows, [])]
            if len(headers) > MAX_COLUMNS:
                raise ValueError(f"Use at most {MAX_COLUMNS} columns.")
            if len(set(filter(None, headers))) != len(list(filter(None, headers))):
                raise ValueError("Column names must be unique.")
            missing = [name for name in required if name not in headers]
            if missing:
                raise ValueError("Missing columns: " + ", ".join(missing) + ".")
            positions = [headers.index(name) for name in required]
            for index, cells in enumerate(rows, start=2):
                if index > MAX_ROWS + 1:
                    raise ValueError(f"Import at most {MAX_ROWS:,} rows per file.")
                if len(cells) > MAX_COLUMNS:
                    raise ValueError(f"Row {index}: Use at most {MAX_COLUMNS} columns.")
                if not any(value is not None and str(value).strip() for value in cells):
                    continue
                row = {
                    name: str(cells[position]).strip()
                    if position < len(cells) and cells[position] is not None else ""
                    for name, position in zip(required, positions)
                }
                try:
                    row = validate_row(kind, row)
                    identifier = row[required[0]]
                    if identifier in identifiers:
                        raise ValueError(f"{required[0]} appears more than once.")
                    if kind == "measurements":
                        blast = (row["site_id"], row["operation_type"], row["location_identifier"], row["blast_date"])
                        if blasts.setdefault(row["blast_id"], blast) != blast:
                            raise ValueError("Rows for the same blast_id must have matching blast details.")
                    identifiers.add(identifier)
                    records.append(row)
                except ValueError as exc:
                    errors.append(f"Row {index}: {exc}")
    except (BadZipFile, InvalidFileException, csv.Error, KeyError, OSError, SyntaxError, UnicodeError):
        errors.append("The file could not be read. Save it as UTF-8 CSV or a standard .xlsx workbook.")
    except ValueError as exc:
        errors.append(str(exc))
    if not records and not errors:
        errors.append("Add at least one data row below the headers.")
    return records, errors


def save_import(kind: str, database: Path, records: list[dict], filename: str) -> tuple[int, int]:
    db = performance_db if kind == "measurements" else diamond_db
    db.init_db(str(database))
    added = 0
    with closing(db.get_connection(str(database))) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        for row in records:
            if kind == "measurements":
                added += insert_measurement(connection, row, filename)
            else:
                added += connection.execute("""
                    INSERT INTO diamond_observations (
                        observation_id, blast_id, sample_lot_id, observation_date, stage,
                        stone_size_class, stone_count, total_carats, damage_type, observer_id,
                        observation_method, stone_quality_category, damage_attribution_cause, attribution_confidence
                    ) VALUES (
                        :observation_id, :blast_id, :sample_lot_id, :observation_date, :stage,
                        :stone_size_class, :stone_count, :total_carats, :damage_type, :observer_id,
                        'unknown', 'unknown', 'unknown', 0
                    ) ON CONFLICT(observation_id) DO NOTHING
                    """, row).rowcount
    return added, len(records) - added


def insert_measurement(connection, row: dict, filename: str) -> int:
    if connection.execute(
        "SELECT 1 FROM actual_measurements WHERE measurement_id = ?", (row["measurement_id"],)
    ).fetchone():
        return 0
    connection.execute("""
        INSERT INTO blast_records (
            blast_id, site_id, operation_type, location_identifier, blast_date, created_at, status
        ) VALUES (
            :blast_id, :site_id, :operation_type, :location_identifier, :blast_date, :created_at, 'MEASURED'
        ) ON CONFLICT(blast_id) DO NOTHING
        """, dict(row, created_at=datetime.now(timezone.utc).isoformat()))
    existing = connection.execute(
        "SELECT site_id, operation_type, location_identifier FROM blast_records WHERE blast_id = ?",
        (row["blast_id"],),
    ).fetchone()
    if tuple(existing) != (row["site_id"], row["operation_type"], row["location_identifier"]):
        raise ValueError("An existing blast ID has different site, operation type, or location details. Check the source file.")
    return connection.execute("""
        INSERT INTO actual_measurements (
            measurement_id, blast_id, measurement_timestamp, target_metric, measured_value, unit, source_filename
        ) VALUES (
            :measurement_id, :blast_id, :measurement_timestamp, :target_metric, :measured_value, :unit, :filename
        )
        """, dict(row, filename=filename)).rowcount
