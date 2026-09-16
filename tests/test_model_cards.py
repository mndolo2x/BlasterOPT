"""
Unit tests for Model Cards generator module (src/model_cards.py).
"""

import os
import tempfile
import pytest
from sklearn.ensemble import RandomForestRegressor
from src.model_cards import generate_model_card


def test_generate_model_card_creates_file_with_non_zero_size():
    """Test generate_model_card creates a Markdown file with non-zero size at the expected path."""
    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        card_path = generate_model_card(
            model_name="Jwaneng_Fragmentation_Predictor",
            model=rf,
            training_data={"size": "120 blast logs", "source": "Jwaneng Open-Pit Mine"},
            performance_metrics={
                "fragmentation": {"R2": 0.956, "RMSE": 12.45, "MAE": 8.10},
                "vibration": {"R2": 0.930, "RMSE": 0.380, "MAE": 0.302},
            },
            version="1.0.0",
            output_dir=tmp_dir,
        )

        assert os.path.exists(card_path), "Model card Markdown file should exist on disk"
        file_size = os.path.getsize(card_path)
        assert file_size > 0, "Model card file should have non-zero size"
        assert card_path.endswith("jwaneng_fragmentation_predictor_v1.0.0.md")

        with open(card_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# 📋 Model Card: Jwaneng_Fragmentation_Predictor" in content
        assert "v1.0.0" in content
        assert "0.956" in content
        assert "7-year regulatory retention period" in content
        assert "Cap. 44:02" in content
