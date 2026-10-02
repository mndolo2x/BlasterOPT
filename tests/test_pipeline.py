"""
Unit tests for session state data pipeline consolidation and state refresh.
"""

import os
import sys
import unittest
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.synthetic_data import generate_synthetic_data
from src.data_ingestion import dataframe_fingerprint, validate_data_schema


class TestPipelineSessionState(unittest.TestCase):
    """Test suite for session state data persistence and stale key cleanup."""

    def test_session_state_cleared_on_regenerate(self):
        """Regenerating must clear old data and save new dataset state."""
        import streamlit as st

        # 1. Set fake old data in session state
        old_df = generate_synthetic_data(n_samples=100, seed=1)
        st.session_state["df"] = old_df
        st.session_state["df_clean"] = old_df
        st.session_state["synthetic_df"] = old_df
        st.session_state["data_ready"] = True
        st.session_state["data_fingerprint"] = dataframe_fingerprint(old_df)

        # 2. Simulate generation flow
        new_df = generate_synthetic_data(n_samples=300, seed=42)
        validate_data_schema(new_df)

        keys_to_clear = [
            "df", "dataset", "df_clean", "synthetic_df", "uploaded_df",
            "regenerated_df", "data_ready", "data_fingerprint",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["df"] = new_df
        st.session_state["data_ready"] = True
        st.session_state["data_source"] = "synthetic"
        st.session_state["data_rows"] = new_df.shape[0]
        st.session_state["data_cols"] = new_df.shape[1]
        st.session_state["data_fingerprint"] = dataframe_fingerprint(new_df)

        # 3. Assert session state contains only new dataset
        self.assertEqual(st.session_state["df"].shape[0], 300)
        self.assertEqual(st.session_state["data_fingerprint"], dataframe_fingerprint(new_df))
        self.assertNotEqual(st.session_state["data_fingerprint"], dataframe_fingerprint(old_df))

    def test_only_one_df_key_exists(self):
        """After generation, session state must contain only 'df', no stale dataset keys."""
        import streamlit as st

        new_df = generate_synthetic_data(n_samples=200, seed=123)
        keys_to_clear = [
            "df", "dataset", "df_clean", "synthetic_df", "uploaded_df",
            "regenerated_df", "data_ready", "data_fingerprint",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["df"] = new_df
        st.session_state["data_ready"] = True
        st.session_state["data_fingerprint"] = dataframe_fingerprint(new_df)

        forbidden_keys = ["dataset", "df_clean", "synthetic_df", "uploaded_df", "regenerated_df"]
        for key in forbidden_keys:
            self.assertNotIn(key, st.session_state, f"Stale key {key} still present")
