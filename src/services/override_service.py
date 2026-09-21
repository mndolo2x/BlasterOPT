"""
Constraint Override Service Module for BlastOpt Botswana.

Allows a supervisor (independent from the requester and approver) to authorize temporary relaxation
of operational constraints (e.g. burden/spacing ranges).
Enforces safety rules:
1. Regulatory Protection: CANNOT relax regulatory limits (PPV, airblast overpressure).
2. Role Check: Only SUPERVISOR role can authorize constraint overrides.
3. Separation of Duties: Supervisor cannot be the requester or the approver.
4. Expiration: Overrides strictly expire after 24 hours.
5. High-Severity Audit: Writes high-severity audit log entries via AuditService.
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

try:
    from src.services.audit_service import AuditService
except ImportError:
    from .audit_service import AuditService

logger = logging.getLogger(__name__)

OVERRIDES_FILE_PATH = "data/processed/overrides.json"
OVERRIDE_STORE: Dict[str, Dict[str, Any]] = {}

REGULATORY_CONSTRAINTS = [
    "max_ppv",
    "max_ppv_mms",
    "max_ppv_mm_s",
    "max_airblast",
    "max_airblast_dbl",
    "max_airblast_db",
    "ppv_limit",
    "airblast_limit",
]


class ConstraintOverride(BaseModel):
    """
    Temporary operational constraint relaxation authorized by an independent supervisor.
    Expires 24 hours after creation.
    """
    override_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID identifier")
    design_id: str = Field(..., description="Target pattern design ID")
    requested_by: str = Field(..., description="User ID of engineer requesting override")
    authorized_by: str = Field(..., description="User ID of supervisor authorizing override")
    user_role: str = Field(..., description="Supervisor role")
    constraint_name: str = Field(..., description="Name of operational constraint (e.g., burden_max_m)")
    original_limit: float = Field(..., description="Original limit before relaxation")
    relaxed_limit: float = Field(..., description="Temporarily relaxed limit value")
    reason: str = Field(..., description="Mandatory written justification for override")
    signature: str = Field(..., description="SHA-256 digital signature hash of override event")
    created_at: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when override was authorized"
    )
    expires_at: Union[datetime, str] = Field(
        default_factory=lambda: (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        description="Timestamp when override expires (24 hours after creation)"
    )


def _load_override_store() -> Dict[str, Dict[str, Any]]:
    """Loads overrides from disk store into memory buffer."""
    global OVERRIDE_STORE
    if os.path.exists(OVERRIDES_FILE_PATH):
        try:
            with open(OVERRIDES_FILE_PATH, "r", encoding="utf-8") as f:
                OVERRIDE_STORE = json.load(f)
        except Exception as err:
            logger.warning(f"Failed to load override store from {OVERRIDES_FILE_PATH}: {err}")
    return OVERRIDE_STORE


def _save_override_store() -> None:
    """Saves override memory buffer to disk file."""
    os.makedirs(os.path.dirname(OVERRIDES_FILE_PATH), exist_ok=True)
    try:
        with open(OVERRIDES_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(OVERRIDE_STORE, f, indent=2)
    except Exception as err:
        logger.error(f"Failed to save override store: {err}")


def compute_override_signature(
    design_id: str,
    constraint_name: str,
    supervisor_id: str,
    created_at: str,
) -> str:
    """Computes SHA-256 digital signature hash for constraint override event."""
    payload = f"{design_id}:{constraint_name}:{supervisor_id}:{created_at}:OVERRIDE_SECRET_SALT"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def authorize_override(
    design_id: str,
    constraint_name: str,
    original_limit: float,
    relaxed_limit: float,
    requester_id: str,
    approver_id: str,
    supervisor_id: str,
    supervisor_role: str = "PIT_SUPERVISOR",
    reason: str = "",
) -> ConstraintOverride:
    """
    Authorizes a temporary 24-hour relaxation of an operational constraint.

    Enforced Rules:
    ---------------
    1. Regulatory protection: Cannot relax regulatory limits (PPV, airblast overpressure).
    2. Role check: Only SUPERVISOR role can authorize constraint overrides.
    3. Separation of duties: Supervisor cannot be requester or approver.
    4. Reason check: Written justification is mandatory.
    5. High-severity audit event logged.
    """
    # 1. Regulatory protection rule
    norm_constraint = constraint_name.lower().strip()
    if norm_constraint in REGULATORY_CONSTRAINTS or any(rc in norm_constraint for rc in ["ppv", "airblast"]):
        raise ValueError(
            f"REGULATORY LIMIT PROTECTION VIOLATION — Cannot override regulatory limit '{constraint_name}'. "
            f"Ground vibration (PPV) and airblast limits are statutory requirements under Mines Act Cap 44:02 "
            f"and cannot be relaxed via override workflows."
        )

    # 2. Role check
    role_norm = supervisor_role.upper().replace(" ", "_")
    if "SUPERVISOR" not in role_norm:
        raise PermissionError(
            f"UNAUTHORIZED — User '{supervisor_id}' with role '{supervisor_role}' is not a SUPERVISOR. "
            f"Constraint overrides require PIT_SUPERVISOR or MINE_SUPERVISOR credentials."
        )

    # 3. Separation of duties
    if supervisor_id == requester_id or supervisor_id == approver_id:
        raise ValueError(
            f"SEPARATION OF DUTIES VIOLATION — Supervisor '{supervisor_id}' cannot authorize an override "
            f"for a design where they act as the requester ('{requester_id}') or approver ('{approver_id}'). "
            f"Overrides require an independent supervisor."
        )

    # 4. Mandatory reason
    reason_clean = reason.strip()
    if not reason_clean:
        raise ValueError("MANDATORY REASON REQUIRED — A written justification is required to authorize an operational constraint override.")

    now_dt = datetime.now(timezone.utc)
    now_ts = now_dt.isoformat()
    exp_ts = (now_dt + timedelta(hours=24)).isoformat()
    sig = compute_override_signature(design_id, constraint_name, supervisor_id, now_ts)

    override = ConstraintOverride(
        override_id=str(uuid.uuid4()),
        design_id=design_id,
        requested_by=requester_id,
        authorized_by=supervisor_id,
        user_role=supervisor_role,
        constraint_name=constraint_name,
        original_limit=original_limit,
        relaxed_limit=relaxed_limit,
        reason=reason_clean,
        signature=sig,
        created_at=now_ts,
        expires_at=exp_ts,
    )

    # Save to disk store
    store = _load_override_store()
    store[override.override_id] = override.model_dump()
    _save_override_store()

    # 5. High-severity audit event
    AuditService().log_event(
        event_type="HIGH_SEVERITY_CONSTRAINT_OVERRIDE",
        user_id=supervisor_id,
        payload=override.model_dump(),
        design_id=design_id,
    )

    logger.warning(
        f"HIGH-SEVERITY OVERRIDE: Constraint '{constraint_name}' relaxed from {original_limit} to {relaxed_limit} "
        f"for design '{design_id}' by supervisor '{supervisor_id}' (Expires: {exp_ts})."
    )

    return override


def is_override_active(override_input: Union[str, ConstraintOverride, Dict[str, Any]]) -> bool:
    """
    Checks if a constraint override is active (i.e. has not passed its 24-hour expiration).
    """
    if isinstance(override_input, ConstraintOverride):
        exp_at = override_input.expires_at
    elif isinstance(override_input, dict):
        exp_at = override_input.get("expires_at")
    else:
        store = _load_override_store()
        dict_data = store.get(str(override_input))
        if not dict_data:
            return False
        exp_at = dict_data.get("expires_at")

    if not exp_at:
        return False

    try:
        if isinstance(exp_at, datetime):
            exp_dt = exp_at if exp_at.tzinfo else exp_at.replace(tzinfo=timezone.utc)
        else:
            exp_dt = datetime.fromisoformat(str(exp_at))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)

        now_dt = datetime.now(timezone.utc)
        return now_dt < exp_dt
    except Exception as err:
        logger.error(f"Error checking override expiration time '{exp_at}': {err}")
        return False
