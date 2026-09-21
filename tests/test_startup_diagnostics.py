"""
Unit tests for Startup Diagnostics subsystem (`src/startup_diagnostics.py`).
"""

import pytest
from src.startup_diagnostics import run_startup_diagnostics, SUBSYSTEM_MODULES


def test_run_startup_diagnostics_returns_structured_summary():
    """Verify run_startup_diagnostics returns expected summary structure and keys."""
    res = run_startup_diagnostics()

    assert isinstance(res, dict)
    assert "overall_status" in res
    assert res["overall_status"] in ["healthy", "degraded", "critical"]
    assert "modules_checked_count" in res
    assert res["modules_checked_count"] == len(SUBSYSTEM_MODULES)
    assert "modules_passed_count" in res
    assert "modules_failed_count" in res
    assert "module_statuses" in res
    assert isinstance(res["module_statuses"], dict)


def test_startup_diagnostics_handles_mock_module_failure(monkeypatch):
    """Verify run_startup_diagnostics gracefully catches import errors and reports degraded status."""
    import importlib
    orig_import = importlib.import_module

    def mock_import(name, package=None):
        if name == "src.fake_broken_subsystem":
            raise ImportError("Mock subsystem failed to load due to missing dependency")
        return orig_import(name, package)

    monkeypatch.setattr("importlib.import_module", mock_import)
    monkeypatch.setattr("src.startup_diagnostics.SUBSYSTEM_MODULES", SUBSYSTEM_MODULES + ["src.fake_broken_subsystem"])

    res = run_startup_diagnostics()

    assert res["overall_status"] == "degraded"
    assert res["modules_failed_count"] >= 1
    assert "src.fake_broken_subsystem" in res["module_statuses"]

    failed_info = res["module_statuses"]["src.fake_broken_subsystem"]
    assert failed_info["status"] == "failed"
    assert "Mock subsystem failed to load" in failed_info["error"]
    assert failed_info["traceback"] is not None
