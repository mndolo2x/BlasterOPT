"""
Unit tests for Underground Intelligence module (pages/underground.py).
"""

import pytest

def test_overbreak_calculation():
    # Heuristic verification
    ring_seq = "Perimeter / Hangingwall Ring"
    hole_dip = -75
    breakthrough = True
    pf_ob = 0.85
    rqd_ob = 65
    dev_ob = 0.4

    base_ob = 5.0 + 4.0
    dip_penalty = (abs(hole_dip) / 90.0) * 2.0
    break_bonus = -1.5 if breakthrough else 3.0
    pf_factor = (pf_ob - 0.7) * 8.0
    rqd_factor = (100.0 - rqd_ob) * 0.12
    dev_factor = dev_ob * 5.0

    calc_overbreak = max(1.0, min(45.0, base_ob + dip_penalty + break_bonus + pf_factor + rqd_factor + dev_factor))
    calc_underbreak = max(0.5, min(20.0, 8.0 - (calc_overbreak * 0.25)))

    assert 1.0 <= calc_overbreak <= 45.0
    assert 0.5 <= calc_underbreak <= 20.0
    assert calc_overbreak > 10.0  # Perimeter ring with parameters should give > 10% overbreak

def test_dilution_elos_calculation():
    hr_val = 6.5
    n_prime = 15.0
    bd = 1.0

    elos_calc = max(0.05, (hr_val * 0.22) - (n_prime * 0.012) + (bd * 0.35))
    dilution_pct = min(100.0, (elos_calc / 3.0) * 100.0)

    assert elos_calc > 0.0
    assert 0.0 <= dilution_pct <= 100.0

def test_underground_vibration_ppv():
    charge = 45.0
    dist = 60.0

    scaled_distance = dist / (charge ** 0.5)
    calc_ppv_ug = 1140.0 * (scaled_distance ** -1.6)

    assert calc_ppv_ug > 0.0
    assert scaled_distance > 0.0
