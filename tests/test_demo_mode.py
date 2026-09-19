"""
Unit tests for DEMO_MODE configuration, provenance metadata, and strict mode enforcement.
"""

import os
import pytest
from src.config import get_demo_mode, require_real_data, get_provenance_badge
from src.mwd_ingestion import connect_to_mqtt
from src.drill_connectivity import connect_to_sandvik, connect_to_epiroc
from src.integrations import connect_to_sap, connect_to_deswik, connect_to_surpac


def test_get_demo_mode_default(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert get_demo_mode() is True


def test_get_demo_mode_false(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    assert get_demo_mode() is False


def test_require_real_data_demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    # Should not raise
    require_real_data("Test Stream")


def test_require_real_data_non_demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError) as exc_info:
        require_real_data("Test Stream")
    assert "REAL DATA REQUIRED" in str(exc_info.value)


def test_get_provenance_badge():
    badge_demo = get_provenance_badge(source="synthetic")
    assert badge_demo["Source"] == "synthetic"

    badge_real = get_provenance_badge(source="field_device", quality="measured")
    assert badge_real["Source"] == "field_device"
    assert badge_real["Data Quality"] == "measured"


def test_connect_to_mqtt_strict_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError):
        connect_to_mqtt("broker.invalid.local")


def test_connect_to_sandvik_strict_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError):
        connect_to_sandvik()


def test_connect_to_epiroc_strict_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError):
        connect_to_epiroc()


def test_connect_to_sap_strict_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError):
        connect_to_sap()


def test_connect_to_deswik_strict_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError):
        connect_to_deswik()


def test_connect_to_surpac_strict_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    with pytest.raises(RuntimeError):
        connect_to_surpac()
