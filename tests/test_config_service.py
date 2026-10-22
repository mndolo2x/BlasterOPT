"""
Unit tests for Centralized Site Configuration Service (config_service.py).
"""

import os
import pytest
import yaml
from src.services.config_service import load_site_config, validate_site_config, get_limit
from src.agent.guardrails import check_regulatory_compliance
from src.agent.tool_registry import BlastDesign
from src.services.guardrail_audit_service import log_guardrail_audit_event, GUARDRAIL_JSONL_PATH


def test_load_valid_yaml():
    cfg = load_site_config(site_id="jwaneng")
    assert cfg["site_id"] == "jwaneng"
    assert cfg["config_version"] == "1.0.0"
    assert "limits" in cfg
    assert get_limit("max_ppv_mm_s") == 5.0
    assert get_limit("max_airblast") == 120.0
    assert get_limit("max_flyrock_distance_m") == 50.0


def test_missing_limit_raises_value_error(tmp_path):
    invalid_yaml = tmp_path / "invalid_limits.yaml"
    invalid_data = {
        "site_id": "test_site",
        "config_version": "1.0.0",
        "status": "approved",
        "limits": {
            "max_ppv_mm_s": {"value": 5.0}
            # Missing max_airblast and max_flyrock_distance_m
        }
    }
    with open(invalid_yaml, "w") as f:
        yaml.dump(invalid_data, f)

    with pytest.raises(ValueError) as excinfo:
        get_limit("max_airblast", site_id="test_site", config_path=str(invalid_yaml))
    assert "missing" in str(excinfo.value).lower()


def test_unverified_config_flagging():
    cfg = load_site_config(site_id="jwaneng")
    assert cfg["is_verified"] is False
    assert "unverified example limits" in cfg["unverified_warning"]


def test_agent_guardrail_warning_on_unverified_config():
    frag = {"d80_cm": 35.0, "d50_cm": 22.0, "distribution": []}
    vib = {"ppv_mm_s": 4.2, "distance_m": 450.0}
    air = {"airblast_db": 114.0, "distance_m": 450.0}
    down = {"crusher_throughput_tph": 2400.0, "specific_energy_kwh_t": 4.2, "dig_rate_tph": 2200.0, "total_cost_per_tonne": 4.80}

    design = BlastDesign(
        design_id="TEST_DESIGN_001",
        bench_id="BENCH_01",
        holes=[],
        powder_factor=0.65,
        burden_m=6.0,
        spacing_m=7.0,
        stemming_m=5.0,
        timing_sequence=[],
        predicted_fragmentation=frag,
        predicted_vibration=vib,
        predicted_airblast=air,
        predicted_downstream=down,
    )

    res = check_regulatory_compliance(design)
    assert "WARNING" in res.reason or "WARNING" in res.safe_response
    assert "unverified example limits" in res.reason or "unverified example limits" in res.safe_response


def test_audit_jsonl_logging(tmp_path):
    test_jsonl = tmp_path / "test_guardrail_log.jsonl"
    record = log_guardrail_audit_event(
        site_id="jwaneng",
        config_version="1.0.0",
        baseline_design_id="BASE_101",
        candidate_design_id="CAND_202",
        action="TEST_ACTION",
        filepath=str(test_jsonl)
    )

    assert os.path.exists(test_jsonl)
    with open(test_jsonl, "r", encoding="utf-8") as f:
        line = f.readline()
        assert "jwaneng" in line
        assert "1.0.0" in line
        assert "BASE_101" in line
        assert "CAND_202" in line
