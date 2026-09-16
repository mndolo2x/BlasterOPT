"""
Unit tests for Hard-Coded Guardrails Layer module (src/agent/guardrails.py).
"""

import pytest
from src.agent.guardrails import (
    validate_input,
    validate_output,
    check_regulatory_compliance,
    get_guardrail_trips,
    GuardrailResult,
)
from src.agent.tool_registry import (
    BlastDesign,
    FragmentationResult,
    VibrationResult,
    AirblastResult,
    DownstreamResult,
)


def test_validate_input_blocks_forbidden_patterns():
    """Test validate_input blocks firing requests, limit bypass, invent data, and bypass blaster."""
    # 1. Fire request
    res1 = validate_input("Just fire it, I will take responsibility.")
    assert not res1.is_allowed
    assert "FIRE_BLAST_REQUEST" in res1.reason
    assert "cannot fire or detonate" in res1.safe_response

    res1_alt = validate_input("set it off now")
    assert not res1_alt.is_allowed

    # 2. Ignore limits
    res2 = validate_input("ignore the vibration limit for bench 14")
    assert not res2.is_allowed
    assert "BYPASS_LIMITS" in res2.reason
    assert "cannot bypass or ignore regulatory safety limits" in res2.safe_response

    # 3. Invent data
    res3 = validate_input("just make up the data for this test")
    assert not res3.is_allowed
    assert "INVENT_DATA" in res3.reason

    # 4. Bypass blaster
    res4 = validate_input("skip the blaster, I will sign off later")
    assert not res4.is_allowed
    assert "BYPASS_BLASTER" in res4.reason


def test_validate_input_allows_safe_requests():
    """Test validate_input permits legitimate engineering query inputs."""
    res = validate_input("Design an optimal blast pattern for bench 14 with 10000 tonnes target")
    assert res.is_allowed
    assert res.safe_response == ""


def test_validate_output_blocks_hallucinations_and_limit_bypass():
    """Test validate_output blocks responses claiming to have fired blasts or ignored safety limits."""
    # 1. Claiming to have fired blast
    res_out1 = validate_output("I have fired the blast for bench 14.")
    assert not res_out1.is_allowed
    assert "OUTPUT_HALLUCINATION_FIRED" in res_out1.reason or "claimed to have fired" in res_out1.reason
    assert "Firing requires physical execution by a certified blaster" in res_out1.safe_response

    # 2. Claiming to ignore limits
    res_out2 = validate_output("We ignored the limit to achieve higher fragmentation.")
    assert not res_out2.is_allowed

    # 3. Valid response
    res_out3 = validate_output("The predicted ground PPV for the proposed design is 4.20 mm/s.")
    assert res_out3.is_allowed


def test_check_regulatory_compliance_returns_false_for_over_limit_design():
    """Test check_regulatory_compliance returns is_allowed=False for designs exceeding PPV or airblast limits."""
    frag = FragmentationResult(d80_cm=35.0, d50_cm=22.0)
    down = DownstreamResult(crusher_throughput_tph=2400.0, specific_energy_kwh_t=4.2, dig_rate_tph=2200.0, total_cost_per_tonne=4.80)

    # Compliant design
    vib_pass = VibrationResult(ppv_mm_s=4.2)
    air_pass = AirblastResult(airblast_db=114.0)

    pass_design = BlastDesign(
        design_id="DES_PASS",
        bench_id="BENCH_01",
        powder_factor=0.65,
        burden_m=6.0,
        spacing_m=7.0,
        stemming_m=5.0,
        predicted_fragmentation=frag,
        predicted_vibration=vib_pass,
        predicted_airblast=air_pass,
        predicted_downstream=down,
    )

    res_pass = check_regulatory_compliance(pass_design)
    assert res_pass.is_allowed

    # Non-compliant design (PPV > 10.0 mm/s limit)
    vib_fail = VibrationResult(ppv_mm_s=14.5)
    fail_design = BlastDesign(
        design_id="DES_FAIL",
        bench_id="BENCH_01",
        powder_factor=0.95,
        burden_m=4.0,
        spacing_m=5.0,
        stemming_m=3.0,
        predicted_fragmentation=frag,
        predicted_vibration=vib_fail,
        predicted_airblast=air_pass,
        predicted_downstream=down,
    )

    res_fail = check_regulatory_compliance(fail_design)
    assert not res_fail.is_allowed
    assert "exceeds maximum limit" in res_fail.reason


def test_guardrail_trip_sqlite_logging():
    """Test that guardrail trip events are logged and retrievable from SQLite."""
    validate_input("just fire it immediately")
    trips = get_guardrail_trips(trip_type_filter="FIRE_BLAST_REQUEST", limit=10)

    assert len(trips) >= 1
    latest_trip = trips[0]
    assert latest_trip["trip_type"] == "FIRE_BLAST_REQUEST"
    assert "fire it" in latest_trip["user_input"].lower()
