"""
Unit tests for Measure-While-Drilling (MWD) ingestion module.
"""

import pytest
import json
from src.mwd_ingestion import parse_mwd_message, connect_to_mqtt


def test_parse_mwd_message_json_string():
    """Test parse_mwd_message correctly parses JSON string payloads."""
    raw_json = json.dumps({
        "hole_id": "BH_102",
        "depth_m": 15.5,
        "penetration_rate_m_hr": 42.0,
        "torque_nm": 1450.0,
        "weight_on_bit_kg": 9200.0,
        "rpm": 120.0,
        "air_pressure_bar": 7.2,
        "vibration_mm_s": 3.1,
        "rock_type": "Granite_Hard",
    })

    parsed = parse_mwd_message(raw_json)

    assert isinstance(parsed, dict)
    assert parsed["hole_id"] == "BH_102"
    assert parsed["depth_m"] == 15.5
    assert parsed["penetration_rate_m_hr"] == 42.0
    assert parsed["torque_nm"] == 1450.0
    assert parsed["weight_on_bit_kg"] == 9200.0
    assert parsed["rpm"] == 120.0
    assert parsed["air_pressure_bar"] == 7.2
    assert parsed["vibration_mm_s"] == 3.1
    assert parsed["rock_type"] == "Granite_Hard"
    assert "specific_energy_mj_m3" in parsed


def test_parse_mwd_message_fallback_defaults():
    """Test parse_mwd_message handles empty/malformed inputs with fallback defaults."""
    parsed = parse_mwd_message("invalid json string")

    assert isinstance(parsed, dict)
    assert "hole_id" in parsed
    assert "depth_m" in parsed
    assert "penetration_rate_m_hr" in parsed
    assert "torque_nm" in parsed
    assert "weight_on_bit_kg" in parsed


def test_connect_to_mqtt_graceful_error_handling():
    """Test connect_to_mqtt handles unreachable broker addresses without crashing."""
    # Attempt connecting to invalid unreachable address
    client = connect_to_mqtt(broker_address="invalid.unreachable.broker.address.local", port=1883)

    # Should return client object attempting async reconnect or None, without raising an unhandled exception
    assert client is None or hasattr(client, "loop_start")
