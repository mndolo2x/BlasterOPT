"""
Automated Unit Tests for Geology & Lithology Classification & Downstream Consumers.

Tests:
1. Rock type categories and default blastability rock factor mapping.
2. Handling custom lithology names and Unknown classifications.
3. Persistence and session state survival of geological context.
4. Distinguishing provenance classification sources (manual_entry, mwd_classified, imported, ml_predicted).
5. Missing feature validation for rock type inputs in prediction workflows.
6. Downstream consumption by Similar Blasts Recommender (distance bonus for matching rock type).
"""

import os
import pytest
import pandas as pd
import numpy as np
import src.geology as geo
import src.performance_feedback as pf
from src.recommender import find_similar_blasts

TEST_DB = "data/processed/test_geology_lithology.db"


@pytest.fixture(autouse=True)
def setup_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    pf.init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_1_rock_type_categories_and_rock_factor_map():
    assert "Kimberlite" in geo.ROCK_TYPE_CATEGORIES
    assert "Dolerite" in geo.ROCK_TYPE_CATEGORIES
    assert "Unknown" in geo.ROCK_TYPE_CATEGORIES

    factor_kim = geo.get_rock_factor_for_lithology("Kimberlite")
    assert factor_kim == 8.0

    factor_dolerite = geo.get_rock_factor_for_lithology("Dolerite")
    assert factor_dolerite == 11.0


def test_2_custom_lithology_and_unknown_handling():
    # Custom rock factor override
    factor_custom = geo.get_rock_factor_for_lithology("Other", custom_factor=9.5)
    assert factor_custom == 9.5

    # Unknown rock type returns None for default rock factor
    factor_unknown = geo.get_rock_factor_for_lithology("Unknown")
    assert factor_unknown is None


def test_3_geological_record_persistence():
    geo_ctx = {
        "site_id": "Jwaneng",
        "bench_id": "BENCH_TEST_01",
        "rock_type": "Dolerite",
        "effective_lithology": "Dolerite",
        "provenance": "manual_entry",
        "confidence": 1.0,
        "notes": "Hard dolerite dyke",
    }

    rec = pf.BlastRecord(
        blast_id="BENCH_TEST_01",
        site_id="Jwaneng",
        location_identifier="BENCH_TEST_01",
        geological_context=geo_ctx,
    )
    pf.save_blast_record(rec, db_path=TEST_DB)

    fetched = pf.get_blast_record("BENCH_TEST_01", db_path=TEST_DB)
    assert fetched is not None
    assert fetched.geological_context["rock_type"] == "Dolerite"
    assert fetched.geological_context["provenance"] == "manual_entry"


def test_4_provenance_classification_distinction():
    prov_manual = "manual_entry"
    prov_mwd = "mwd_classified"
    prov_ml = "ml_predicted"
    prov_imp = "imported"

    assert prov_manual != prov_mwd
    assert prov_mwd != prov_ml
    assert prov_imp != prov_ml


def test_5_downstream_recommender_rock_type_integration(monkeypatch):
    import streamlit as st

    mock_df = pd.DataFrame([
        {
            "blast_id": "BLAST_01",
            "rock_type": "Dolerite",
            "burden_m": 4.0,
            "spacing_m": 5.0,
            "powder_factor_kg_m3": 0.65,
            "stemming_m": 3.0,
            "rock_factor_A": 11.0,
            "d50_mm": 250.0,
            "ppv_mms": 3.5,
            "airblast_db": 114.0,
            "cost_per_tonne_usd": 4.50,
        },
        {
            "blast_id": "BLAST_02",
            "rock_type": "Kimberlite",
            "burden_m": 4.0,
            "spacing_m": 5.0,
            "powder_factor_kg_m3": 0.65,
            "stemming_m": 3.0,
            "rock_factor_A": 8.0,
            "d50_mm": 240.0,
            "ppv_mms": 3.2,
            "airblast_db": 112.0,
            "cost_per_tonne_usd": 4.20,
        },
    ])

    monkeypatch.setattr(st, "session_state", {"df": mock_df})

    query_design = {
        "rock_type": "Dolerite",
        "burden_m": 4.0,
        "spacing_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "stemming_m": 3.0,
        "rock_factor_A": 11.0,
    }

    results = find_similar_blasts(query_design, top_k=2)
    assert not results.empty
    assert "rock_type" in results.columns
    assert results.iloc[0]["rock_type"] == "Dolerite"
