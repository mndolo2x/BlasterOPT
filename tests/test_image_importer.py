"""
Unit tests for ImageAnalysisImporter in src/fragmentation/image_analysis.py.
"""

import os
import pytest
import pandas as pd
from src.fragmentation import ImageAnalysisImporter


def test_image_analysis_importer_fit_and_d80():
    df = pd.DataFrame({
        "size_mm": [10.0, 50.0, 100.0, 200.0, 300.0, 500.0],
        "percent_passing": [5.0, 20.0, 40.0, 65.0, 80.0, 95.0],
    })

    importer = ImageAnalysisImporter()
    d80 = importer.calculate_d80_from_image(df)
    assert d80 == 300.0

    fits = importer.fit_distributions_to_measured(df)
    assert "best_fit" in fits
    assert "swebrec" in fits
    assert "rosin_rammler" in fits
    assert "lognormal" in fits
