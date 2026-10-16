"""
Unit tests for Digital Twin wrapper and scene template rendering.
"""

import os
import pytest
from pathlib import Path
from src.ui.digital_twin.render import (
    load_scene_html,
    get_scene_template_size,
    TEMPLATE_PATH,
)


def test_scene_template_exists():
    """Verify scene_template.html exists at the expected path."""
    assert TEMPLATE_PATH.exists(), f"Missing template at {TEMPLATE_PATH}"


def test_scene_template_size():
    """Verify scene_template.html is non-empty and has full content."""
    assert get_scene_template_size() >= 20000


def test_load_scene_html_returns_string():
    """Verify load_scene_html returns a non-empty string."""
    html = load_scene_html()
    assert isinstance(html, str)
    assert len(html) > 0


def test_load_scene_html_contains_doctype():
    """Verify raw HTML starts with DOCTYPE tag."""
    html = load_scene_html().strip()
    assert html.lower().startswith("<!doctype html")


def test_load_scene_html_contains_webgl_canvas():
    """Verify scene template contains WebGL canvas element and context initialization."""
    html = load_scene_html()
    assert 'id="gl"' in html
    assert "getContext('webgl'" in html or "getContext(\"webgl\"" in html


def test_load_scene_html_raises_when_missing(tmp_path, monkeypatch):
    """Verify load_scene_html raises FileNotFoundError if template missing."""
    fake_path = tmp_path / "non_existent_template.html"
    monkeypatch.setattr("src.ui.digital_twin.render.TEMPLATE_PATH", fake_path)
    with pytest.raises(FileNotFoundError):
        load_scene_html()


def test_digital_twin_page_imports_load_scene_html():
    """Verify pages/digital_twin.py imports load_scene_html."""
    page_path = Path("pages/digital_twin.py")
    assert page_path.exists()
    content = page_path.read_text(encoding="utf-8")
    assert "from src.ui.digital_twin.render import load_scene_html" in content
    assert "components.html(" in content
