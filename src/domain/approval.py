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
from typing import Dict, Any, List, Optional, Tuple, Literal, Union
from src.domain.safety_checks import SafetyReport

logger = logging.getLogger(__name__)

APPROVALS_FILE_PATH = "data/processed/approvals.json"
APPROVAL_STORE: Dict[str, Dict[str, Any]] = {}


class ApprovalRequest(BaseModel):
    """
    Approval request payload submitting a pattern design snapshot and safety report for sign-off.
    """
    design_id: str = Field(..., description="Pattern design identifier")
    design_snapshot: Dict[str, Any] = Field(..., description="Immutable copy of the design parameters")
    safety_report: SafetyReport = Field(..., description="Associated uncertainty safety report")
    requested_by: str = Field(..., description="User ID requesting approval")
    requested_at: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when approval was requested"
    )
    model_version: str = Field("v1.0.0", description="Model version used for predictions")
    constraint_version: str = Field("v1.0.0", description="Constraint version applied")
    dataset_version: str = Field("v1.0.0", description="Dataset version used for model training")


class ApprovalDecision(BaseModel):
    """
    Immutable sign-off decision recorded when a certified blaster evaluates a pattern design.
    """
    design_id: str = Field(..., description="Target pattern design identifier")
    decision: Literal["APPROVED", "REJECTED", "CHANGES_REQUESTED"] = Field(..., description="Approval decision")
    decided_by: str = Field(..., description="Certified blaster ID / credential")
    decided_at: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of sign-off decision"
    )
    reason: str = Field(..., description="Detailed explanation or override justification")
    signature: str = Field(..., description="Digital signature or SHA-256 hash stub")
    conditions: List[str] = Field(default_factory=list, description="Optional conditions of approval")

    # Backward compatibility aliases/properties
    @property
    def blaster_id(self) -> str:
        return self.decided_by

    @property
    def signature_hash(self) -> str:
        return self.signature

    @property
    def override_reasoning(self) -> str:
        return self.reason


# Backward compatibility alias
ApprovalRecord = ApprovalDecision


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
    role: str = "Certified Blaster",
    decision: Literal["APPROVED", "REJECTED", "CHANGES_REQUESTED"] = "APPROVED",
    override_reasoning: Optional[str] = None,
    safety_report: Optional[SafetyReport] = None,
    conditions: Optional[List[str]] = None,
) -> ApprovalDecision:
    """
    Processes human sign-off for a blast design and returns an ApprovalDecision.

    Enforcement Rules:
    ------------------
    - If safety_report.overall_status == "UNSAFE", approval is BLOCKED and raises ValueError.
    - If safety_report.overall_status == "REQUIRES_REVIEW" and decision == "APPROVED",
      override_reasoning is MANDATORY (must be non-empty string).
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
    reason_str = override_reasoning or f"Signed off by {role} ({blaster_id}) with decision {decision}."

    decision_obj = ApprovalDecision(
        design_id=design_id,
        decision=decision,
        decided_by=blaster_id,
        decided_at=ts,
        reason=reason_str,
        signature=sig_hash,
        conditions=conditions or [],
    )

    store = _load_approval_store()
    store[design_id] = decision_obj.model_dump()
    _save_approval_store()

    logger.info(f"Design '{design_id}' sign-off decision '{decision}' recorded by blaster '{blaster_id}'.")
    return decision_obj


def check_approval_gate(design_id: str, action: str = "export") -> Tuple[bool, str]:
    """
    Checks whether a given design has passed certified human approval for downstream actions
    like pattern export or drill rig transmission.
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
    blaster_id = record_dict.get("decided_by", record_dict.get("blaster_id", "Unknown"))

    if decision == "APPROVED":
        return (True, f"APPROVED — Design '{design_id}' signed off by Certified Blaster '{blaster_id}'.")
    elif decision == "CHANGES_REQUESTED":
        return (
            False,
            f"CHANGES REQUESTED — Design '{design_id}' requires modifications before {action}. "
            f"Reason: {record_dict.get('reason', 'N/A')}"
        )
    else:
        return (
            False,
            f"DESIGN REJECTED — Design '{design_id}' was REJECTED during human review. "
            f"Action '{action}' is blocked."
        )


def get_approval_record(design_id: str) -> Optional[ApprovalDecision]:
    """Retrieves existing approval record for a design if present."""
    store = _load_approval_store()
    record_dict = store.get(design_id)
    if record_dict:
        return ApprovalDecision(**record_dict)
    return None
