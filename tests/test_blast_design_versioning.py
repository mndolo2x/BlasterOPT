"""
Unit tests for immutable blast design versioning (src/domain/blast_design.py).
"""

import pytest
from src.domain.safety_checks import SafetyReport
from src.domain.blast_design import (
    BlastDesign,
    BlastDesignVersion,
    compute_content_hash,
    create_next_version,
)


def test_blast_design_model():
    design = BlastDesign(
        burden_m=6.0,
        spacing_m=7.0,
        stemming_m=5.0,
        powder_factor_kg_m3=0.65,
    )
    assert design.burden_m == 6.0
    assert design.spacing_m == 7.0
    assert design.bench_height_m == 12.0


def test_compute_content_hash_determinism():
    design_a = BlastDesign(burden_m=6.0, spacing_m=7.0, stemming_m=5.0, powder_factor_kg_m3=0.65)
    design_b = BlastDesign(burden_m=6.0, spacing_m=7.0, stemming_m=5.0, powder_factor_kg_m3=0.65)

    hash_a = compute_content_hash(design_a)
    hash_b = compute_content_hash(design_b)

    assert len(hash_a) == 64  # SHA-256 hex string
    assert hash_a == hash_b  # Deterministic equal hash for identical parameters

    # Differing parameters produce different hash
    design_c = BlastDesign(burden_m=6.5, spacing_m=7.0, stemming_m=5.0, powder_factor_kg_m3=0.65)
    hash_c = compute_content_hash(design_c)
    assert hash_c != hash_a


def test_create_version_and_progression():
    safety_rep_safe = SafetyReport(
        overall_status="SAFE",
        checks=[],
        requires_engineer_review=False,
        blocks_export=False,
    )

    design_v1_data = {
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
    }

    v1 = BlastDesignVersion.create(
        design_id="PATTERN_JWA_2026_01",
        design_data=design_v1_data,
        safety_report=safety_rep_safe,
        created_by="ENGINEER_ALICE",
        version=1,
        approval_status="APPROVED",
    )

    assert v1.design_id == "PATTERN_JWA_2026_01"
    assert v1.version == 1
    assert v1.parent_version is None
    assert v1.approval_status == "APPROVED"
    assert len(v1.content_hash) == 64

    # Now modify approved v1 design to create v2
    design_v2_data = {
        "burden_m": 6.5,  # Modified burden
        "spacing_m": 7.5,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.60,
    }

    v2 = create_next_version(
        current_version=v1,
        new_design_data=design_v2_data,
        new_safety_report=safety_rep_safe,
        created_by="ENGINEER_BOB",
    )

    assert v2.design_id == "PATTERN_JWA_2026_01"
    assert v2.version == 2
    assert v2.parent_version == 1  # Points to parent version 1
    assert v2.created_by == "ENGINEER_BOB"
    assert v2.approval_status == "DRAFT"  # Reset to DRAFT for review
    assert v2.content_hash != v1.content_hash  # New hash generated
    assert v1.content_hash == compute_content_hash(v1.design)  # v1 hash remains unchanged
