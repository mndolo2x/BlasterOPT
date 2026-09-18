"""
Data Ingestion Module for Seismograph Vibration Readings.
"""

import json
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Union, Optional
from src.site_calibration.models import SeismographReading, IngestionReport

logger = logging.getLogger(__name__)


def validate_reading_dict(row: Dict[str, Any]) -> Tuple[bool, Optional[SeismographReading], Optional[str]]:
    """
    Validates a raw dictionary against SeismographReading constraints.

    Validation rules:
    - distance_m > 0
    - ppv_mm_s > 0
    - charge_per_delay_kg > 0
    - gsi_of_transmission_strata in [0, 100] if provided
    """
    try:
        dist = float(row.get("distance_m", 0.0))
        ppv = float(row.get("ppv_mm_s", 0.0))
        charge = float(row.get("charge_per_delay_kg", 0.0))

        if dist <= 0:
            return False, None, f"Invalid distance_m ({dist}): must be > 0"
        if ppv <= 0:
            return False, None, f"Invalid ppv_mm_s ({ppv}): must be > 0"
        if charge <= 0:
            return False, None, f"Invalid charge_per_delay_kg ({charge}): must be > 0"

        gsi_val = row.get("gsi_of_transmission_strata")
        if gsi_val is not None and gsi_val != "" and not pd.isna(gsi_val):
            gsi_float = float(gsi_val)
            if gsi_float < 0 or gsi_float > 100:
                return False, None, f"Invalid gsi ({gsi_float}): must be between 0 and 100"
            row["gsi_of_transmission_strata"] = gsi_float
        else:
            row["gsi_of_transmission_strata"] = None

        blast_id = str(row.get("blast_id", "BLAST_UNKNOWN")).strip()
        site_id = str(row.get("site_id", "SITE_DEFAULT")).strip()

        reading = SeismographReading(
            timestamp=str(row.get("timestamp")) if row.get("timestamp") else None,
            blast_id=blast_id,
            site_id=site_id,
            distance_m=dist,
            ppv_mm_s=ppv,
            charge_per_delay_kg=charge,
            dominant_frequency_hz=float(row["dominant_frequency_hz"]) if row.get("dominant_frequency_hz") else None,
            gsi_of_transmission_strata=row["gsi_of_transmission_strata"],
            rock_type=str(row.get("rock_type", "Kimberlite")),
        )
        return True, reading, None
    except Exception as e:
        return False, None, f"Parsing error: {e}"


def load_seismograph_data(
    source: Union[str, List[Dict[str, Any]], pd.DataFrame]
) -> Tuple[pd.DataFrame, IngestionReport]:
    """
    Loads seismograph readings from CSV filepath, JSON filepath, DataFrame, or List[Dict].
    Validates required fields and deduplicates by (blast_id, distance_m).

    Returns:
    --------
    Tuple[pd.DataFrame, IngestionReport]
        DataFrame of accepted valid readings + IngestionReport summarizing results.
    """
    raw_records = []

    if isinstance(source, str):
        if source.endswith(".csv"):
            df_raw = pd.read_csv(source)
            raw_records = df_raw.to_dict(orient="records")
        elif source.endswith(".json"):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_records = data
                elif isinstance(data, dict) and "readings" in data:
                    raw_records = data["readings"]
        else:
            raise ValueError(f"Unsupported file format for source: {source}")
    elif isinstance(source, pd.DataFrame):
        raw_records = source.to_dict(orient="records")
    elif isinstance(source, list):
        raw_records = source
    else:
        raise TypeError(f"Unsupported data source type: {type(source)}")

    accepted_readings: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    seen_keys = set()

    for idx, row in enumerate(raw_records):
        is_valid, reading, reason = validate_reading_dict(row)
        if not is_valid or reading is None:
            rejected_rows.append({"row_index": idx, "row_data": row, "reason": reason})
            continue

        # Deduplicate by (blast_id, distance_m)
        dedup_key = (reading.blast_id, round(reading.distance_m, 2))
        if dedup_key in seen_keys:
            rejected_rows.append({"row_index": idx, "row_data": row, "reason": f"Duplicate reading for blast_id '{reading.blast_id}' at distance {reading.distance_m}m"})
            continue

        seen_keys.add(dedup_key)
        accepted_readings.append(reading.model_dump())

    df_accepted = pd.DataFrame(accepted_readings) if accepted_readings else pd.DataFrame(columns=[
        "timestamp", "blast_id", "site_id", "distance_m", "ppv_mm_s",
        "charge_per_delay_kg", "dominant_frequency_hz", "gsi_of_transmission_strata", "rock_type"
    ])

    report = IngestionReport(
        accepted_count=len(accepted_readings),
        rejected_count=len(rejected_rows),
        rejected_rows=rejected_rows,
    )

    return df_accepted, report
