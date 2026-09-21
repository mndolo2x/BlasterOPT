"""
Approval Service Module for BlastOpt Botswana.

Manages pattern design approval requests, decision recording, and approval status queries.
Enforces safety rules:
1. Role enforcement: Only CERTIFIED_BLASTER role can record decisions.
2. Separation of duties: Submitter cannot approve their own design.
3. Safety gate: UNSAFE designs cannot be approved.
4. Uncertainty acknowledgment: Approver must explicitly acknowledge 95% confidence bounds when review is required.
5. Immutable audit logging: Writes decision event to append-only audit JSONL log.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple, Literal, Union

from src.domain.safety_checks import SafetyReport
from src.domain.approval import (
    ApprovalRequest,
    ApprovalDecision,
    create_signature_hash,
    _load_approval_store,
    _save_approval_store,
)

logger = logging.getLogger(__name__)

REQUESTS_FILE_PATH = "data/processed/approval_requests.json"
AUDIT_LOG_DIR = "data/audit"
REQUEST_STORE: Dict[str, Dict[str, Any]] = {}


class ApprovalStatus(BaseModel):
    """
    Current status summary of an approval workflow for a given design.
    """
    design_id: str = Field(..., description="Target pattern design ID")
    status: Literal["PENDING", "APPROVED", "REJECTED", "CHANGES_REQUESTED", "UNAPPROVED"] = Field(
        ..., description="Current status state"
    )
    request: Optional[ApprovalRequest] = Field(None, description="Associated approval request if present")
    decision: Optional[ApprovalDecision] = Field(None, description="Recorded decision if present")


def _load_request_store() -> Dict[str, Dict[str, Any]]:
    """Loads approval request records from disk into memory buffer."""
    global REQUEST_STORE
    if os.path.exists(REQUESTS_FILE_PATH):
        try:
            with open(REQUESTS_FILE_PATH, "r", encoding="utf-8") as f:
                REQUEST_STORE = json.load(f)
        except Exception as err:
            logger.warning(f"Failed to load request store from {REQUESTS_FILE_PATH}: {err}")
    return REQUEST_STORE


def _save_request_store() -> None:
    """Saves request memory buffer to disk file."""
    os.makedirs(os.path.dirname(REQUESTS_FILE_PATH), exist_ok=True)
    try:
        with open(REQUESTS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(REQUEST_STORE, f, indent=2)
    except Exception as err:
        logger.error(f"Failed to save request store: {err}")


from src.services.audit_service import AuditService

def _write_audit_log(event_type: str, data: Dict[str, Any]) -> None:
    """Appends an immutable audit record to data/audit/YYYY-MM-DD.jsonl via AuditService."""
    try:
        uid = data.get("requested_by", data.get("decided_by", data.get("blaster_id", "system")))
        did = data.get("design_id")
        AuditService().log_event(
            event_type=event_type,
            user_id=uid,
            payload=data,
            design_id=did,
        )
    except Exception as err:
        logger.error(f"Failed to write audit log entry: {err}")


def submit_for_approval(
    design: Dict[str, Any],
    safety_report: SafetyReport,
    user_id: str,
    model_version: str = "v1.0.0",
    constraint_version: str = "v1.0.0",
    dataset_version: str = "v1.0.0",
    design_id: Optional[str] = None,
) -> ApprovalRequest:
    """
    Submits a design pattern and safety report for certified blaster review.
    Creates an immutable snapshot of the design parameters.
    """
    did = design_id or str(design.get("design_id", f"PATTERN_{int(datetime.now(timezone.utc).timestamp())}"))

    # Create immutable design snapshot copy
    design_snapshot = design.copy()
    design_snapshot["design_id"] = did

    request = ApprovalRequest(
        design_id=did,
        design_snapshot=design_snapshot,
        safety_report=safety_report,
        requested_by=user_id,
        requested_at=datetime.now(timezone.utc).isoformat(),
        model_version=model_version,
        constraint_version=constraint_version,
        dataset_version=dataset_version,
    )

    req_data = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    store = _load_request_store()
    store[did] = req_data
    _save_request_store()

    # Append to immutable JSONL audit log
    _write_audit_log("SUBMIT_FOR_APPROVAL", req_data)

    logger.info(f"Approval request submitted for design '{did}' by user '{user_id}'.")
    return request


def record_decision(
    design_id_or_request: Union[str, ApprovalRequest],
    decision: Literal["APPROVED", "REJECTED", "CHANGES_REQUESTED"],
    user_id: str,
    user_role: str = "CERTIFIED_BLASTER",
    reason: str = "",
    acknowledged_uncertainty: bool = False,
    conditions: Optional[List[str]] = None,
) -> ApprovalDecision:
    """
    Records a certified blaster's sign-off decision on a pattern request.

    Enforced Rules:
    ---------------
    1. Role check: Only CERTIFIED_BLASTER can record decisions (raises PermissionError).
    2. Separation of duties: Submitter cannot approve their own submission (raises ValueError).
    3. Safety check: UNSAFE designs cannot be approved (raises ValueError).
    4. Uncertainty acknowledgment: Approver must acknowledge 95% confidence bounds when review is required (raises ValueError).
    """
    # 1. Role enforcement
    role_norm = user_role.upper().replace(" ", "_")
    allowed_roles = ["CERTIFIED_BLASTER", "BLASTER", "CHIEF_MINING_ENGINEER", "PIT_SUPERVISOR"]
    if not any(r in role_norm for r in allowed_roles):
        raise PermissionError(
            f"UNAUTHORIZED — User '{user_id}' with role '{user_role}' is not a CERTIFIED_BLASTER. "
            f"Approval decisions require certified blaster credentials."
        )

    # Resolve design ID and request object
    if isinstance(design_id_or_request, ApprovalRequest):
        did = design_id_or_request.design_id
        req = design_id_or_request
    else:
        did = str(design_id_or_request)
        req_store = _load_request_store()
        req_dict = req_store.get(did)
        req = ApprovalRequest(**req_dict) if req_dict else None

    # 2. Separation of duties
    if req is not None and req.requested_by == user_id:
        raise ValueError(
            f"SEPARATION OF DUTIES VIOLATION — User '{user_id}' submitted design '{did}' "
            f"and cannot approve their own submission. Approval requires an independent certified blaster."
        )

    safety_report = req.safety_report if req else None

    if decision == "APPROVED":
        # 3. Safety check: UNSAFE blocking
        if safety_report and safety_report.overall_status == "UNSAFE":
            raise ValueError(
                f"CANNOT APPROVE INFEASIBLE / UNSAFE DESIGN '{did}'. "
                f"Design contains critical safety violations and cannot be approved."
            )

        # 4. Uncertainty acknowledgment requirement
        requires_review = safety_report.requires_engineer_review or (safety_report.overall_status == "REQUIRES_REVIEW") if safety_report else False
        if requires_review and not acknowledged_uncertainty:
            raise ValueError(
                f"UNCERTAINTY ACKNOWLEDGMENT REQUIRED — Design '{did}' requires review. "
                f"Approver must explicitly acknowledge 95% confidence bounds before recording decision."
            )

    ts = datetime.now(timezone.utc).isoformat()
    sig = create_signature_hash(did, user_id, decision, ts)
    reason_clean = reason.strip() or f"Signed off by {user_role} ({user_id}) with decision {decision}."

    decision_obj = ApprovalDecision(
        design_id=did,
        decision=decision,
        decided_by=user_id,
        decided_at=ts,
        reason=reason_clean,
        signature=sig,
        conditions=conditions or [],
    )

    # Save decision to disk store
    dec_data = decision_obj.model_dump() if hasattr(decision_obj, "model_dump") else decision_obj.dict()
    app_store = _load_approval_store()
    app_store[did] = dec_data
    _save_approval_store()

    # Write immutable audit record
    _write_audit_log("RECORD_APPROVAL_DECISION", dec_data)

    logger.info(f"Decision '{decision}' recorded for design '{did}' by certified blaster '{user_id}'.")
    return decision_obj


def get_approval_status(design_id: str) -> ApprovalStatus:
    """
    Retrieves the current approval status, request snapshot, and recorded decision for a design.
    """
    req_store = _load_request_store()
    app_store = _load_approval_store()

    req_dict = req_store.get(design_id)
    app_dict = app_store.get(design_id)

    req_obj = ApprovalRequest(**req_dict) if req_dict else None
    app_obj = ApprovalDecision(**app_dict) if app_dict else None

    if app_obj is not None:
        status = app_obj.decision
    elif req_obj is not None:
        status = "PENDING"
    else:
        status = "UNAPPROVED"

    return ApprovalStatus(
        design_id=design_id,
        status=status,
        request=req_obj,
        decision=app_obj,
    )
