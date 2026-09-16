"""
Unit tests for model card generation module (src/model_cards.py).
"""

import os
import tempfile
import pytest
from sklearn.ensemble import RandomForestRegressor
from src.model_cards import generate_model_card


def test_generate_model_card_creates_non_empty_file():
    """Verifies that generate_model_card creates a Markdown file with non-zero size."""
    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        card_path = generate_model_card(
            model_name="Test Fragmentation Regressor",
            model=rf,
            training_data={"size": "100 samples", "source": "Test Mine"},
            performance_metrics={"R2": 0.91, "RMSE": 10.2},
            output_dir=tmp_dir,
        )

        assert os.path.exists(card_path), "Model card file should exist"
        file_size = os.path.getsize(card_path)
        assert file_size > 0, "Model card file should have non-zero size"

        with open(card_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# 📋 Model Card: Test Fragmentation Regressor" in content
        assert "0.91" in content
        assert "Cap. 44:02" in content
