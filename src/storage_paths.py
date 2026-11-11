"""Default paths for the historical SQLite databases."""

from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PERFORMANCE_DB_PATH = PROCESSED_DATA_DIR / "blast_performance.db"
DIAMOND_DB_PATH = PROCESSED_DATA_DIR / "diamond_preservation.db"
