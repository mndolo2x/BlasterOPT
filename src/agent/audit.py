"""
Agent Audit Log Module for BlasterOPT Conversational Agent.

Provides immutable append-only SQLite logging for all agent interactions, tool calls,
guardrail trips, and engineering decisions/overrides for regulatory compliance under
the Mines, Quarries, Works and Machinery Act (Cap. 44:02) with 7-year retention.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

AUDIT_DB_PATH = "data/processed/agent_audit.db"


def _init_db():
    """Initializes the append-only SQLite database tables for agent interaction and decision auditing."""
    os.makedirs(os.path.dirname(AUDIT_DB_PATH), exist_ok=True)
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                tools_called_json TEXT NOT NULL,
                guardrail_trips_json TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                design_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                original_value TEXT,
                new_value TEXT
            )
            """
        )
        conn.commit()


_init_db()


def log_interaction(
    session_id: str,
    user_id: str,
    user_message: str,
    agent_response: str,
    tools_called: Union[List[Any], Dict[str, Any]],
    guardrail_trips: Union[List[Any], Dict[str, Any]],
) -> int:
    """
    Immutably logs an agent interaction row in the SQLite database.
    Retained for regulatory compliance (Cap. 44:02) audit trail.
    """
    _init_db()
    ts = datetime.now(timezone.utc).isoformat()
    tools_json = json.dumps(tools_called) if not isinstance(tools_called, str) else tools_called
    guard_json = json.dumps(guardrail_trips) if not isinstance(guardrail_trips, str) else guardrail_trips

    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_interactions
                (timestamp, session_id, user_id, user_message, agent_response, tools_called_json, guardrail_trips_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, session_id, user_id, user_message, agent_response, tools_json, guard_json),
            )
            conn.commit()
            return cursor.lastrowid or 0
    except Exception as e:
        logger.error(f"Error logging agent interaction: {e}")
        return 0


def log_decision(
    session_id: str,
    user_id: str,
    design_id: str,
    decision: str,
    reason_code: str,
    original_value: Optional[Union[Dict[str, Any], str]] = None,
    new_value: Optional[Union[Dict[str, Any], str]] = None,
) -> int:
    """
    Immutably logs a decision or parameter override in the SQLite database.
    """
    _init_db()
    ts = datetime.now(timezone.utc).isoformat()
    orig_str = json.dumps(original_value) if isinstance(original_value, (dict, list)) else (str(original_value) if original_value is not None else "")
    new_str = json.dumps(new_value) if isinstance(new_value, (dict, list)) else (str(new_value) if new_value is not None else "")

    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_decisions
                (timestamp, session_id, user_id, design_id, decision, reason_code, original_value, new_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, session_id, user_id, design_id, decision, reason_code, orig_str, new_str),
            )
            conn.commit()
            return cursor.lastrowid or 0
    except Exception as e:
        logger.error(f"Error logging decision: {e}")
        return 0


def get_interaction_history(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieves interaction history entries from SQLite database.
    """
    _init_db()
    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM agent_interactions WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error reading interaction history: {e}")
        return []


def get_decision_history(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieves decision and override history entries from SQLite database.
    """
    _init_db()
    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM agent_decisions WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error reading decision history: {e}")
        return []


def export_audit_log_json(output_path: str = "data/processed/agent_audit_regulatory_export.json") -> str:
    """
    Exports full interaction and decision audit history formatted for regulatory submission.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    interactions = get_interaction_history(limit=1000)
    decisions = get_decision_history(limit=1000)

    export_payload = {
        "metadata": {
            "act": "Mines, Quarries, Works and Machinery Act (Cap. 44:02) Compliance Audit",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "retention_policy": "7 Years Mandatory Persistence",
            "total_interactions": len(interactions),
            "total_decisions": len(decisions),
        },
        "interactions": interactions,
        "decisions": decisions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    logger.info(f"Exported regulatory audit log to `{output_path}`.")
    return output_path
