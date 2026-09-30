"""
Unit tests for Fragmentation Analysis page rendering.
"""

import pytest
from src.fragmentation.fragmentation_analysis import render_fragmentation_analysis_page


def test_fragmentation_analysis_import():
    assert callable(render_fragmentation_analysis_page)
