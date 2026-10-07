"""
Unit tests for the Similar Blast Recommender module.
"""

import pytest
import pandas as pd
import numpy as np
import streamlit as st
from src.recommender import find_similar_blasts
from src.synthetic_data import generate_synthetic_blast_data


@pytest.fixture(autouse=True)
def setup_session_state():
    """Setup st.session_state['df'] before each test."""
    st.session_state["df"] = generate_synthetic_blast_data(num_samples=50, seed=42)
    yield
    st.session_state.clear()


def test_find_similar_returns_correct_count():
    """Test that find_similar_blasts returns top_k rows."""
    query = {"burden_m": 4.5, "spacing_m": 5.5, "powder_factor_kg_m3": 0.60, "stemming_m": 3.5, "rock_factor_A": 7.0}

    res = find_similar_blasts(query, top_k=5)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 5

    res_10 = find_similar_blasts(query, top_k=10)
    assert len(res_10) == 10


def test_find_similar_returns_sorted_by_distance():
    """Test that similarity distances in returned DataFrame are non-negative and sorted ascending."""
    query = {"burden_m": 4.0, "spacing_m": 5.0, "powder_factor_kg_m3": 0.55}
    res = find_similar_blasts(query, top_k=5)

    distances = res["distance"].values
    assert np.all(distances >= 0.0)
    assert np.all(np.diff(distances) >= 0.0)


def test_results_change_when_query_changes():
    """Test that changing query design parameters alters distance and top match."""
    query1 = {"burden_m": 3.5, "spacing_m": 4.0, "powder_factor_kg_m3": 0.40, "stemming_m": 2.5, "rock_factor_A": 5.0}
    query2 = {"burden_m": 6.5, "spacing_m": 8.0, "powder_factor_kg_m3": 0.90, "stemming_m": 5.5, "rock_factor_A": 10.0}

    res1 = find_similar_blasts(query1, top_k=5)
    res2 = find_similar_blasts(query2, top_k=5)

    assert res1.iloc[0]["blast_id"] != res2.iloc[0]["blast_id"] or res1.iloc[0]["distance"] != res2.iloc[0]["distance"]


def test_empty_dataset_returns_empty_dataframe():
    """Test that find_similar_blasts handles empty or None DataFrame gracefully."""
    query = {"burden_m": 4.5, "spacing_m": 5.5}

    st.session_state["df"] = pd.DataFrame()
    res_empty = find_similar_blasts(query, top_k=5)
    assert isinstance(res_empty, pd.DataFrame)
    assert len(res_empty) == 0

    st.session_state["df"] = None
    res_none = find_similar_blasts(query, top_k=5)
    assert isinstance(res_none, pd.DataFrame)
    assert len(res_none) == 0


def test_results_contain_expected_columns():
    """Test that output DataFrame contains all expected display columns including compliant."""
    query = {"burden_m": 4.5, "spacing_m": 5.5}
    res = find_similar_blasts(query, top_k=5)

    expected_cols = [
        "rank", "blast_id", "distance", "burden_m", "spacing_m",
        "powder_factor_kg_m3", "d80_mm", "ppv_mms", "airblast_db", "cost_per_tonne_usd", "compliant"
    ]
    for col in expected_cols:
        assert col in res.columns

    # Verify compliant column logic: True if airblast <= 120 and ppv <= 5.0
    for _, row in res.iterrows():
        expected_compliant = (row["airblast_db"] <= 120.0) and (row["ppv_mms"] <= 5.0)
        assert row["compliant"] == expected_compliant
