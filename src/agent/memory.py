"""
Conversation Memory Manager for BlasterOPT LangGraph Agent.

Supports:
- Redis-backed checkpointer for conversation history (with local memory fallback).
- Session-scoped short-term memory.
- Long-term SQLite memory for user preferences, role defaults, and past interactions.
"""

import os
import sqlite3
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MEMORY_DB_PATH = "data/processed/user_memory.db"


class AgentMemoryManager:
    """
    Manages short-term session state, Redis conversation checkpointer, and long-term user preferences.
    """

    def __init__(self, session_id: str = "DEFAULT_SESSION", redis_url: Optional[str] = None):
        self.session_id = session_id
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.short_term_memory: Dict[str, Any] = {}
        self._init_sqlite_db()

    def _init_sqlite_db(self):
        """Initializes SQLite database for long-term user memory and preferences."""
        os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_language TEXT DEFAULT 'en',
                    user_role TEXT DEFAULT 'engineer',
                    favorite_bench_id TEXT,
                    default_max_ppv REAL DEFAULT 5.0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    agent_summary TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieves long-term user preferences from SQLite database."""
        try:
            with sqlite3.connect(MEMORY_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            logger.error(f"Error reading user preferences: {e}")

        return {
            "user_id": user_id,
            "preferred_language": "en",
            "user_role": "engineer",
            "default_max_ppv": 5.0,
        }

    def save_user_preferences(
        self,
        user_id: str,
        preferred_language: str = "en",
        user_role: str = "engineer",
        favorite_bench_id: Optional[str] = None,
        default_max_ppv: float = 5.0,
    ):
        """Saves or updates long-term user preferences in SQLite."""
        import datetime
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            with sqlite3.connect(MEMORY_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_preferences (user_id, preferred_language, user_role, favorite_bench_id, default_max_ppv, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        preferred_language=excluded.preferred_language,
                        user_role=excluded.user_role,
                        favorite_bench_id=excluded.favorite_bench_id,
                        default_max_ppv=excluded.default_max_ppv,
                        updated_at=excluded.updated_at
                    """,
                    (user_id, preferred_language, user_role, favorite_bench_id, default_max_ppv, ts),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")

    def update_short_term(self, key: str, value: Any):
        """Updates short-term session memory key-value pair."""
        self.short_term_memory[key] = value

    def get_short_term(self, key: str, default: Any = None) -> Any:
        """Retrieves value from short-term session memory."""
        return self.short_term_memory.get(key, default)
