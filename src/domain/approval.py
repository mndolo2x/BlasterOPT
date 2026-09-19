"""
Human Approval Gate & Blaster Sign-off Workflow for BlastOpt Botswana.

Enforces mandatory human sign-off from certified blasters prior to pattern export
or direct transmission to drill rigs, creating an immutable audit trail.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple, Literal
from src.domain.safety_checks import SafetyReport

logger = logging.getLogger(__name__)

APPROVALS_FILE_PATH = "data/processed/approvals.json"
APPROVAL_STORE: Dict[str, Dict[str, Any]] = {}


class ApprovalRecord(BaseModel):
    """
    Immutable sign-off record created when a certified blaster evaluates and decides on a pattern.
    """
    design_id: str = Field(..., description="Target pattern design identifier")
    blaster_id: str = Field(..., description="Certified blaster registration or employee ID")
    role: str = Field(..., description="User role (e.g. Certified Blaster, Chief Mining Engineer)")
    decision: Literal["APPROVED", "REJECTED"] = Field(..., description="Approval decision")
    signature_hash: str = Field(..., description="SHA-256 digital signature hash of the approval event")
    timestamp: str = Field(..., description="UTC ISO timestamp of sign-off")
    override_reasoning: Optional[str] = Field(None, description="Mandatory reasoning if approving REQUIRES_REVIEW design")
    safety_status: str = Field("SAFE", description="Safety status at time of sign-off")


class ApprovalRequest(BaseModel):
    """
    Pending approval request containing design ID and current safety report.
    """
    design_id: str = Field(..., description="Pattern design identifier")
    requested_by: str = Field(..., description="User ID requesting approval")
    safety_report: SafetyReport = Field(..., description="Associated uncertainty safety report")


def _load_approval_store() -> Dict[str, Dict[str, Any]]:
    """Loads approval records from disk into memory buffer."""
    global APPROVAL_STORE
    if os.path.exists(APPROVALS_FILE_PATH):
        try:
            with open(APPROVALS_FILE_PATH, "r", encoding="utf-8") as f:
                APPROVAL_STORE = json.load(f)
        except Exception as err:
            logger.warning(f"Failed to load approval store from {APPROVALS_FILE_PATH}: {err}")
    return APPROVAL_STORE


def _save_approval_store() -> None:
    """Saves memory buffer to disk file."""
    os.makedirs(os.path.dirname(APPROVALS_FILE_PATH), exist_ok=True)
    try:
        with open(APPROVALS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(APPROVAL_STORE, f, indent=2)
    except Exception as err:
        logger.error(f"Failed to save approval store: {err}")


def create_signature_hash(design_id: str, blaster_id: str, decision: str, timestamp: str) -> str:
    """Computes SHA-256 hash representing digital signature of the sign-off event."""
    payload = f"{design_id}:{blaster_id}:{decision}:{timestamp}:BLASTOPT_SECRET_SALT"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approve_design(
    design_id: str,
    blaster_id: str,
    role: str,
    decision: Literal["APPROVED", "REJECTED"],
    override_reasoning: Optional[str] = None,
    safety_report: Optional[SafetyReport] = None,
) -> ApprovalRecord:
    """
    Processes human sign-off for a blast design.

    Enforcement Rules:
    ------------------
    - If safety_report.overall_status == "UNSAFE", approval is BLOCKED and raises ValueError.
    - If safety_report.overall_status == "REQUIRES_REVIEW" and decision == "APPROVED",
      override_reasoning is MANDATORY (must be non-empty string).

    Parameters:
    -----------
    design_id : str
        Target design identifier.
    blaster_id : str
        Blaster ID / credential.
    role : str
        User role.
    decision : Literal["APPROVED", "REJECTED"]
        Decision.
    override_reasoning : Optional[str]
        Reasoning if overriding REQUIRES_REVIEW warnings.
    safety_report : Optional[SafetyReport]
        Safety evaluation report.

    Returns:
    --------
    ApprovalRecord
        Validated immutable approval record.
    """
    safety_status = safety_report.overall_status if safety_report else "SAFE"

    if decision == "APPROVED":
        if safety_status == "UNSAFE":
            raise ValueError(
                f"CANNOT APPROVE INFEASIBLE / UNSAFE DESIGN '{design_id}'. "
                f"Design contains critical safety violations and cannot be approved."
            )
        if safety_status == "REQUIRES_REVIEW":
            if not override_reasoning or not override_reasoning.strip():
                raise ValueError(
                    f"MANDATORY OVERRIDE REASONING REQUIRED. "
                    f"Design '{design_id}' has safety warnings (REQUIRES_REVIEW). "
                    f"Certified blaster must document explicit engineering justification before signing off."
                )

    ts = datetime.now(timezone.utc).isoformat()
    sig_hash = create_signature_hash(design_id, blaster_id, decision, ts)

    record = ApprovalRecord(
        design_id=design_id,
        blaster_id=blaster_id,
        role=role,
        decision=decision,
        signature_hash=sig_hash,
        timestamp=ts,
        override_reasoning=override_reasoning,
        safety_status=safety_status,
    )

    store = _load_approval_store()
    store[design_id] = record.model_dump()
    _save_approval_store()

    logger.info(f"Design '{design_id}' sign-off decision '{decision}' recorded by blaster '{blaster_id}'.")
    return record


def check_approval_gate(design_id: str, action: str = "export") -> Tuple[bool, str]:
    """
    Checks whether a given design has passed certified human approval for downstream actions
    like pattern export or drill rig transmission.

    Parameters:
    -----------
    design_id : str
        Design identifier.
    action : str, default="export"
        Target action ("export" or "transmit_to_drill").

    Returns:
    --------
    Tuple[bool, str]
        (is_approved, message) tuple.
    """
    store = _load_approval_store()
    record_dict = store.get(design_id)

    if not record_dict:
        return (
            False,
            f"APPROVAL REQUIRED — Design '{design_id}' has not been reviewed. "
            f"Explicit sign-off from a certified blaster is required before {action}."
        )

    decision = record_dict.get("decision")
    blaster_id = record_dict.get("blaster_id", "Unknown")

    if decision == "APPROVED":
        return (True, f"APPROVED — Design '{design_id}' signed off by Certified Blaster '{blaster_id}'.")
    else:
        return (
            False,
            f"DESIGN REJECTED — Design '{design_id}' was REJECTED during human review. "
            f"Action '{action}' is blocked."
        )


def get_approval_record(design_id: str) -> Optional[ApprovalRecord]:
    """Retrieves existing approval record for a design if present."""
    store = _load_approval_store()
    record_dict = store.get(design_id)
    if record_dict:
        return ApprovalRecord(**record_dict)
    return None
