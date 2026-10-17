"""
Unit tests for Digital Twin loader and page.
"""

import importlib.util
from pathlib import Path
from src.ui.digital_twin.loader import load_and_inject_scene


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


def test_load_and_inject_scene_missing_template(tmp_path, monkeypatch):
    """Verify load_and_inject_scene returns None when scene_template.html is missing."""
    test_data = {"bench_id": "BENCH_14", "powder_factor_kg_m3": 0.65}
    # Point template_path to non-existent location
    fake_path = tmp_path / "scene_template.html"
    monkeypatch.setattr("src.ui.digital_twin.loader.Path", lambda p: fake_path)
    result = load_and_inject_scene(test_data)
    assert result is None


def test_load_and_inject_scene_success(tmp_path, monkeypatch):
    """Verify load_and_inject_scene injects JSON data into HTML template."""
    template_dir = tmp_path / "src" / "ui" / "digital_twin"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_file = template_dir / "scene_template.html"
    template_file.write_text("<!doctype html><html><head></head><body></body></html>", encoding="utf-8")

    def mock_path(path_str):
        if "scene_template.html" in path_str:
            return template_file
        return Path(path_str)

    monkeypatch.setattr("src.ui.digital_twin.loader.Path", mock_path)

    test_data = {"bench_id": "BENCH_14", "bench_height_m": 15.0}
    result = load_and_inject_scene(test_data)
    assert result is not None
    assert "window.BLASTOPT_SCENE_DATA =" in result
    assert '"bench_id": "BENCH_14"' in result
