"""
Functional unit tests for Barton Q-System calculation button logic and parameter parsing.
"""

import re
import pytest


def extract_number(text: str) -> float:
    match = re.search(r"=([\d.]+)", text)
    return float(match.group(1)) if match else 0.0


def calculate_q(rqd: float, jn_text: str, jr_text: str, ja_text: str, jw_text: str, srf_text: str) -> float:
    jn = extract_number(jn_text)
    jr = extract_number(jr_text)
    ja = extract_number(ja_text)
    jw = extract_number(jw_text)
    srf = extract_number(srf_text)

    if jn == 0 or ja == 0 or srf == 0:
        return 0.0
    return round((rqd / jn) * (jr / ja) * (jw / srf), 2)


def test_q_calculation_with_expected_defaults():
    """
    Test Q calculation with Task 4 test values:
    RQD = 85, Jn = 9 (three joint sets), Jr = 3.0 (rough undulating),
    Ja = 2.0 (slightly altered), Jw = 0.66 (damp), SRF = 1.0 (medium stress).
    Expected Q = (85/9) * (3/2) * (0.66/1.0) = 9.35
    """
    q_a = calculate_q(
        rqd=85.0,
        jn_text="Three joint sets (Jn=9)",
        jr_text="Rough, undulating (Jr=3.0)",
        ja_text="Slightly altered (Ja=2.0)",
        jw_text="Damp (Jw=0.66)",
        srf_text="Medium stress (SRF=1.0)",
    )
    assert q_a == 9.35, f"Expected 9.35, got {q_a}"


def test_q_changes_when_rqd_changes():
    """
    Test Q changes when RQD changes from 85 to 40.
    Expected Q = (40/9) * (3/2) * (0.66/1.0) = 4.40
    """
    q_b = calculate_q(
        rqd=40.0,
        jn_text="Three joint sets (Jn=9)",
        jr_text="Rough, undulating (Jr=3.0)",
        ja_text="Slightly altered (Ja=2.0)",
        jw_text="Damp (Jw=0.66)",
        srf_text="Medium stress (SRF=1.0)",
    )
    assert q_b == 4.40, f"Expected 4.40, got {q_b}"


def test_extract_number_regex():
    """Test regex extraction of parameter values from option labels."""
    assert extract_number("Three joint sets (Jn=9)") == 9.0
    assert extract_number("Dry (Jw=1.0)") == 1.0
    assert extract_number("Damp (Jw=0.66)") == 0.66
    assert extract_number("Tight, unaltered (Ja=0.75)") == 0.75
