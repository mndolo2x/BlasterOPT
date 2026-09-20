"""
Immutable Hash-Chained Audit Service for BlastOpt Botswana.

Implements append-only JSONL event logging at `data/audit/YYYY-MM-DD.jsonl`.
Calculates SHA-256 payload hashes and chains previous event hashes (`previous_event_hash`)
so any tampering or deletion is instantly detectable via `verify_chain(date_str)`.
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

AUDIT_DIR = "data/audit"


class AuditEvent(BaseModel):
    """
    Immutable audit event record containing cryptographic hashes for payload and chain integrity.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID event identifier")
    timestamp: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event UTC timestamp"
    )
    event_type: str = Field(..., description="Category of event (e.g. prediction, approval, export, safety_check, override)")
    user_id: str = Field(..., description="User ID or service account triggering event")
    design_id: Optional[str] = Field(None, description="Blast pattern design ID")
    design_version: Optional[int] = Field(None, description="Blast pattern design version number")
    model_version: Optional[str] = Field("v1.0.0", description="Model version active at event time")
    constraint_version: Optional[str] = Field("v1.0.0", description="Constraint version active at event time")
    dataset_version: Optional[str] = Field("v1.0.0", description="Dataset version active at event time")
    payload_hash: str = Field(..., description="SHA-256 hash of event payload dictionary")
    payload: Dict[str, Any] = Field(..., description="Detailed event payload data")
    previous_event_hash: Optional[str] = Field(None, description="SHA-256 hash of previous event in chain for tamper detection")


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hash of a payload dictionary."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_event_hash(event_dict: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hash of a full audit event dictionary."""
    serialized = json.dumps(event_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AuditService:
    """
    Appends audit events to JSONL log files with cryptographic hash-chaining.
    """

    def __init__(self, audit_dir: str = AUDIT_DIR):
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)

    def _get_log_filepath(self, date_str: Optional[str] = None) -> str:
        d_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not d_str.endswith(".jsonl"):
            filename = f"{d_str}.jsonl"
        else:
            filename = d_str
        return os.path.join(self.audit_dir, filename)

    def log_event(
        self,
        event_type: str,
        user_id: str,
        payload: Dict[str, Any],
        design_id: Optional[str] = None,
        design_version: Optional[int] = None,
        model_version: Optional[str] = "v1.0.0",
        constraint_version: Optional[str] = "v1.0.0",
        dataset_version: Optional[str] = "v1.0.0",
        date_str: Optional[str] = None,
    ) -> AuditEvent:
        """
        Logs an audit event to `data/audit/YYYY-MM-DD.jsonl` with previous event hash chaining.
        """
        filepath = self._get_log_filepath(date_str)

        # Compute payload hash
        p_hash = compute_payload_hash(payload)

        # Read previous event to compute previous_event_hash
        previous_event_hash = None
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        last_line_dict = json.loads(lines[-1])
                        previous_event_hash = compute_event_hash(last_line_dict)
            except Exception as err:
                logger.warning(f"Error reading previous event hash from {filepath}: {err}")

        ts = datetime.now(timezone.utc).isoformat()
        evt = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=ts,
            event_type=event_type,
            user_id=user_id,
            design_id=design_id,
            design_version=design_version,
            model_version=model_version,
            constraint_version=constraint_version,
            dataset_version=dataset_version,
            payload_hash=p_hash,
            payload=payload,
            previous_event_hash=previous_event_hash,
        )

        # Append to JSONL file
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt.model_dump(), default=str) + "\n")

        logger.info(f"Logged audit event '{event_type}' (id: {evt.event_id}) to {filepath}.")
        return evt


def verify_chain(date_str: str, audit_dir: str = AUDIT_DIR) -> bool:
    """
    Verifies hash-chain integrity of `data/audit/YYYY-MM-DD.jsonl`.
    Validates:
    1. payload_hash matches SHA-256 of payload for every event.
    2. previous_event_hash matches compute_event_hash of previous event line.

    Returns True if chain is completely intact; False if any line was altered or deleted.
    """
    if not date_str.endswith(".jsonl"):
        filename = f"{date_str}.jsonl"
    else:
        filename = date_str

    filepath = os.path.join(audit_dir, filename)

    if not os.path.exists(filepath):
        logger.warning(f"Audit log file {filepath} does not exist.")
        return True  # Vacuously valid for non-existent/empty day log

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return True

        prev_event_hash_calc = None

        for idx, line in enumerate(lines):
            evt_dict = json.loads(line)

            # 1. Verify payload_hash
            payload = evt_dict.get("payload", {})
            p_hash_calc = compute_payload_hash(payload)
            if evt_dict.get("payload_hash") != p_hash_calc:
                logger.error(f"Tamper detected in {filepath} line {idx}: payload_hash mismatch.")
                return False

            # 2. Verify previous_event_hash chain
            expected_prev_hash = evt_dict.get("previous_event_hash")
            if idx == 0:
                if expected_prev_hash != prev_event_hash_calc:
                    # Line 0 previous_event_hash must be None
                    logger.error(f"Tamper detected in {filepath} line 0: initial previous_event_hash is not None.")
                    return False
            else:
                if expected_prev_hash != prev_event_hash_calc:
                    logger.error(f"Tamper detected in {filepath} line {idx}: previous_event_hash chain broken.")
                    return False

            # Update prev_event_hash_calc for next line
            prev_event_hash_calc = compute_event_hash(evt_dict)

        return True

    except Exception as err:
        logger.error(f"Error during audit chain verification for {filepath}: {err}")
        return False
