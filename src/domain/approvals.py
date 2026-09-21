"""
Re-export module for src/domain/approval.py to provide exact import compatibility for approvals.py.
"""

from src.domain.approval import (
    ApprovalRequest,
    ApprovalDecision,
    ApprovalRecord,
    create_signature_hash,
    approve_design,
    check_approval_gate,
    get_approval_record,
)

__all__ = [
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalRecord",
    "create_signature_hash",
    "approve_design",
    "check_approval_gate",
    "get_approval_record",
]
