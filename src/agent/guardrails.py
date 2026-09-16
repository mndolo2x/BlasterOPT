"""
Guardrail Layer Module (Hard-Coded Safety) for BlasterOPT Conversational Agent.

Enforces deterministic, non-negotiable safety rules that CANNOT be overridden by LLMs, users, or tools:
1. Prevents requests to fire blasts or approve designs without a certified blaster.
2. Prevents bypassing regulatory vibration/airblast limits or inventing data.
3. Performs hallucination and compliance checks on agent outputs.
4. Immutably logs guardrail trip events in a SQLite database.
"""

import os
import re
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.agent.tool_registry import BlastDesign
from src.regulatory import load_regulatory_limits

logger = logging.getLogger(__name__)

DB_PATH = "data/processed/guardrail_trips.db"


class GuardrailResult(BaseModel):
    is_allowed: bool = Field(..., description="Whether the action or response is permitted")
    reason: str = Field(..., description="Explanation of allowed status or guardrail trip reason")
    safe_response: str = Field(..., description="Deterministic safe response to display to user")


# Pre-compiled regex patterns for forbidden user inputs
FORBIDDEN_INPUT_PATTERNS = [
    {
        "type": "FIRE_BLAST_REQUEST",
        "pattern": r"\b(fire\s+it|just\s+fire|set\s+it\s+off|detonate\s+now|trigger\s+blast)\b",
        "reason": "Request to fire or detonate a blast directly.",
        "safe_response": "I cannot fire or detonate a blast. Firing requires a certified blaster physically present at the site. I can prepare the approval package for your certified blaster. Would you like me to do that?",
    },
    {
        "type": "UNAUTHORIZED_APPROVAL",
        "pattern": r"\b(i\s+approve|just\s+approve\s+it|bypass\s+approval|approve\s+without\s+blaster)\b",
        "reason": "Request to approve a design without a certified blaster.",
        "safe_response": "I cannot approve a blast design. Approval requires a certified blaster with active Botswana Department of Mines credentials. I can route this design to the certified blaster for review.",
    },
    {
        "type": "BYPASS_LIMITS",
        "pattern": r"\b(ignore\s+the\s+(vibration|airblast|ppv)\s+limit|exceed\s+the\s+limit|bypass\s+limits|disregard\s+safety)\b",
        "reason": "Request to bypass or ignore regulatory safety limits.",
        "safe_response": "I cannot bypass or ignore regulatory safety limits. All blast designs must comply with Botswana Department of Mines environmental thresholds (PPV <= 10.0 mm/s, Airblast <= 120 dB).",
    },
    {
        "type": "INVENT_DATA",
        "pattern": r"\b(just\s+make\s+up\s+the\s+data|assume\s+the\s+values|fake\s+the\s+data|invent\s+measurements)\b",
        "reason": "Request to invent or fake blast measurement data.",
        "safe_response": "I cannot invent or fake blast measurement data. All inputs must come from validated field logs, MWD drill rig telematics, or site geological models.",
    },
    {
        "type": "BYPASS_BLASTER",
        "pattern": r"\b(skip\s+the\s+blaster|i'll\s+sign\s+off\s+later|override\s+the\s+blaster)\b",
        "reason": "Request to bypass the certified blaster sign-off process.",
        "safe_response": "I cannot bypass the certified blaster sign-off process. Under the Mines, Quarries, Works and Machinery Act (Cap. 44:02), certified blaster review is legally mandated.",
    },
]


def _init_db():
    """Initializes the SQLite database table for immutable guardrail trip logging."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS guardrail_trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trip_type TEXT NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                user_id TEXT DEFAULT 'USER_DEFAULT'
            )
            """
        )
        conn.commit()


_init_db()


def log_guardrail_trip(
    trip_type: str,
    user_input: str,
    agent_response: str,
    user_id: str = "USER_DEFAULT",
):
    """
    Immutably logs a guardrail trip event in the SQLite database.
    """
    _init_db()
    ts = datetime.utcnow().isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO guardrail_trips (timestamp, trip_type, user_input, agent_response, user_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, trip_type, user_input, agent_response, user_id),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error logging guardrail trip: {e}")


def get_guardrail_trips(trip_type_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieves logged guardrail trips from the SQLite database.
    """
    _init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if trip_type_filter and trip_type_filter != "ALL":
                cursor.execute(
                    "SELECT * FROM guardrail_trips WHERE trip_type = ? ORDER BY id DESC LIMIT ?",
                    (trip_type_filter, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM guardrail_trips ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching guardrail trips: {e}")
        return []


def validate_input(user_input: str, user_id: str = "USER_DEFAULT") -> GuardrailResult:
    """
    Validates user input against forbidden patterns (firing requests, limit bypass, invent data, bypass blaster).
    """
    cleaned_input = user_input.strip()

    for item in FORBIDDEN_INPUT_PATTERNS:
        if re.search(item["pattern"], cleaned_input, re.IGNORECASE):
            trip_type = item["type"]
            safe_resp = item["safe_response"]
            reason = item["reason"]

            log_guardrail_trip(
                trip_type=trip_type,
                user_input=cleaned_input,
                agent_response=safe_resp,
                user_id=user_id,
            )

            return GuardrailResult(
                is_allowed=False,
                reason=f"Guardrail trip ({trip_type}): {reason}",
                safe_response=safe_resp,
            )

    return GuardrailResult(
        is_allowed=True,
        reason="Input passed all deterministic guardrail safety checks.",
        safe_response="",
    )


def validate_output(
    agent_response: str,
    tool_results: Dict[str, Any] = {},
    user_id: str = "USER_DEFAULT",
) -> GuardrailResult:
    """
    Validates agent output before sending to user:
    - Hallucination check: Claims to have fired/detonated a blast.
    - Directive check: Presents recommendations as mandatory orders without blaster signoff.
    """
    # 1. Hallucination check - claim to have fired blast
    if re.search(r"\b(i\s+have\s+fired|blast\s+has\s+been\s+fired|detonated\s+the\s+blast|executed\s+detonation)\b", agent_response, re.IGNORECASE):
        safe_resp = "I have generated the optimized blast design parameters. Note: Firing requires physical execution by a certified blaster."
        log_guardrail_trip(
            trip_type="OUTPUT_HALLUCINATION_FIRED",
            user_input="[OUTPUT CHECK]",
            agent_response=agent_response,
            user_id=user_id,
        )
        return GuardrailResult(
            is_allowed=False,
            reason="Agent output claimed to have fired a blast.",
            safe_response=safe_resp,
        )

    # 2. Check if output claims limits were ignored
    if re.search(r"\b(ignored\s+the\s+limit|exceeded\s+safety\s+threshold)\b", agent_response, re.IGNORECASE):
        safe_resp = "The proposed design parameters have been evaluated against Botswana Department of Mines environmental limits."
        log_guardrail_trip(
            trip_type="OUTPUT_LIMIT_BYPASS_CLAIM",
            user_input="[OUTPUT CHECK]",
            agent_response=agent_response,
            user_id=user_id,
        )
        return GuardrailResult(
            is_allowed=False,
            reason="Agent output claimed to ignore safety limits.",
            safe_response=safe_resp,
        )

    return GuardrailResult(
        is_allowed=True,
        reason="Output passed all deterministic guardrail checks.",
        safe_response=agent_response,
    )


def check_regulatory_compliance(
    design: BlastDesign,
    max_cost_limit: Optional[float] = None,
    user_id: str = "USER_DEFAULT",
) -> GuardrailResult:
    """
    Validates a BlastDesign object against environmental limits and max cost bounds.
    """
    limits = load_regulatory_limits()
    max_ppv = float(limits.get("max_ppv_mms", 10.0))
    max_airblast = float(limits.get("max_airblast_dbl", 120.0))

    predicted_ppv = design.predicted_vibration.ppv_mm_s
    predicted_airblast = design.predicted_airblast.airblast_db
    predicted_cost = design.predicted_downstream.total_cost_per_tonne

    violations = []
    if predicted_ppv > max_ppv:
        violations.append(f"Predicted PPV ground vibration ({predicted_ppv:.2f} mm/s) exceeds maximum limit ({max_ppv:.1f} mm/s).")

    if predicted_airblast > max_airblast:
        violations.append(f"Predicted Airblast overpressure ({predicted_airblast:.1f} dB) exceeds maximum limit ({max_airblast:.1f} dB).")

    if max_cost_limit is not None and predicted_cost > max_cost_limit:
        violations.append(f"Predicted unit cost (${predicted_cost:.2f}/t) exceeds specified cost target (${max_cost_limit:.2f}/t).")

    if violations:
        reason_str = " | ".join(violations)
        safe_resp = (
            f"The proposed design '{design.design_id}' violates safety or cost constraints: {reason_str} "
            f"Recommendation: Reduce maximum charge per delay or increase stemming length to bring metrics within limits."
        )

        log_guardrail_trip(
            trip_type="REGULATORY_LIMIT_EXCEEDED",
            user_input=f"[DESIGN CHECK {design.design_id}]",
            agent_response=safe_resp,
            user_id=user_id,
        )

        return GuardrailResult(
            is_allowed=False,
            reason=reason_str,
            safe_response=safe_resp,
        )

    return GuardrailResult(
        is_allowed=True,
        reason=f"Design '{design.design_id}' complies with all regulatory limits (PPV <= {max_ppv} mm/s, Airblast <= {max_airblast} dB).",
        safe_response="",
    )
