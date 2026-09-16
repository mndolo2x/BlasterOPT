"""
Unit tests for the Write-Ahead Log (WAL) and SyncManager offline-first sync module.
"""

import os
import pytest
from src.offline_sync import WriteAheadLog, SyncManager, resolve_conflicts


def test_write_ahead_log_appends_and_persists(tmp_path):
    """Test WriteAheadLog queues actions persistently to disk."""
    wal_file = str(tmp_path / "test_wal.json")
    wal = WriteAheadLog(wal_path=wal_file)

    assert len(wal.get_pending()) == 0

    entry1 = wal.append("FIELD_LOG", {"hole_id": "BH_101", "depth_m": 15.0})
    entry2 = wal.append("DESIGN_UPDATE", {"pattern_id": "DES_002", "burden_m": 6.0})

    assert len(wal.get_pending()) == 2
    assert entry1["action_type"] == "FIELD_LOG"
    assert entry2["action_type"] == "DESIGN_UPDATE"

    # Reload WAL from file to verify persistence
    wal_reloaded = WriteAheadLog(wal_path=wal_file)
    assert len(wal_reloaded.get_pending()) == 2

    # Mark first entry completed
    wal.mark_completed(entry1["wal_id"])
    assert len(wal.get_pending()) == 1
    assert wal.get_pending()[0]["wal_id"] == entry2["wal_id"]


def test_sync_manager_exponential_backoff_delay():
    """Test SyncManager calculates exponential backoff delay correctly."""
    sync_mgr = SyncManager(initial_delay_sec=1.0, backoff_factor=2.0)

    assert sync_mgr.calculate_backoff_delay(0) == 1.0   # 1.0 * (2^0) = 1.0
    assert sync_mgr.calculate_backoff_delay(1) == 2.0   # 1.0 * (2^1) = 2.0
    assert sync_mgr.calculate_backoff_delay(2) == 4.0   # 1.0 * (2^2) = 4.0
    assert sync_mgr.calculate_backoff_delay(3) == 8.0   # 1.0 * (2^3) = 8.0


def test_sync_manager_replays_pending_wal_actions(tmp_path):
    """Test SyncManager replays pending WAL queue when online."""
    wal_file = str(tmp_path / "test_wal_sync.json")
    wal = WriteAheadLog(wal_path=wal_file)
    wal.append("FIELD_LOG", {"hole_id": "BH_105", "depth_m": 15.5})

    sync_mgr = SyncManager(wal=wal, max_retries=2)

    # Replay sync when online
    res = sync_mgr.sync_all(online_check_fn=lambda: True)

    assert res["status"] == "sync_complete"
    assert res["synced_count"] == 1
    assert res["remaining_pending"] == 0
    assert res["last_sync_timestamp"] is not None


def test_resolve_conflicts_strategies():
    """Test resolve_conflicts correctly resolves local vs remote edits using strategies."""
    local_edit = {"hole_id": "BH_101", "depth_m": 15.5, "timestamp": "2026-09-16T10:30:00"}
    remote_edit = {"hole_id": "BH_101", "depth_m": 15.0, "timestamp": "2026-09-16T10:00:00"}

    # Strategy 1: last_write_wins (Local is newer)
    res_lww = resolve_conflicts(local_edit, remote_edit, strategy="last_write_wins")
    assert res_lww["depth_m"] == 15.5
    assert res_lww["conflict_resolved_by"] == "last_write_wins_local"

    # Strategy 2: remote_wins
    res_rw = resolve_conflicts(local_edit, remote_edit, strategy="remote_wins")
    assert res_rw["depth_m"] == 15.0

    # Strategy 3: merge_conservative
    res_merge = resolve_conflicts({"depth_m": 16.0}, {"depth_m": 15.0, "charge_mass_per_hole_kg": 320.0}, strategy="merge_conservative")
    assert res_merge["depth_m"] == 16.0
    assert res_merge["charge_mass_per_hole_kg"] == 320.0
