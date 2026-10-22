"""
Guardrail & Optimization Audit Logger Service.

Appends audit log records containing site_id, config_version, timestamp, baseline_design_id,
and candidate_design_id to data/audit/guardrail_log.jsonl.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

GUARDRAIL_JSONL_PATH = "data/audit/guardrail_log.jsonl"


def log_guardrail_audit_event(
    site_id: str,
    config_version: str,
    baseline_design_id: str,
    candidate_design_id: str,
    action: str = "SAVE_DESIGN_OR_OPTIMIZE",
    additional_data: Optional[Dict[str, Any]] = None,
    filepath: str = GUARDRAIL_JSONL_PATH,
) -> Dict[str, Any]:
    """
    Appends an audit log record to data/audit/guardrail_log.jsonl.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "site_id": site_id,
        "config_version": config_version,
        "action": action,
        "baseline_design_id": baseline_design_id,
        "candidate_design_id": candidate_design_id,
    }

    if additional_data:
        record["additional_data"] = additional_data

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Error writing to guardrail JSONL audit log ({filepath}): {e}")

    return record
