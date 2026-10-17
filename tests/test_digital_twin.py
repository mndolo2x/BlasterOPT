"""
Unit tests for Digital Twin page placeholder.
"""

import importlib.util
from pathlib import Path


def test_digital_twin_page_exists():
    """Verify pages/digital_twin.py exists."""
    page_path = Path("pages/digital_twin.py")
    assert page_path.exists()


def test_digital_twin_page_imports():
    """Verify pages/digital_twin.py parses and imports without syntax errors."""
    page_path = Path("pages/digital_twin.py")
    spec = importlib.util.spec_from_file_location("digital_twin_page", page_path)
    assert spec is not None
    assert spec.loader is not None
