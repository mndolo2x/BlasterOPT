"""
Unit tests for pages/predictor.py and inline predictor removal.
"""
import subprocess


def test_predictor_calls_model_selector():
    """pages/predictor.py must call render_page_model_selector."""
    result = subprocess.run(
        ["grep", "-q", 'render_page_model_selector("predictor")', "pages/predictor.py"],
        capture_output=True,
    )
    assert result.returncode == 0, "pages/predictor.py does not call render_page_model_selector('predictor')"


def test_no_inline_predictor_in_app():
    """app.py predictor block must delegate to pages/predictor.py."""
    result = subprocess.run(
        ["grep", "-n", "predict_single_blast", "app.py"],
        capture_output=True, text=True,
    )
    lines = [line for line in result.stdout.splitlines() if "elif active_module == \"predictor\":" in line or "input_payload" in line]
    assert len(lines) == 0, f"Inline predictor workflow still in app.py:\n{result.stdout}"


def test_predictor_uses_selected_model():
    """pages/predictor.py must use model.predict(X)."""
    result = subprocess.run(
        ["grep", "-q", "model.predict(", "pages/predictor.py"],
        capture_output=True,
    )
    assert result.returncode == 0, "pages/predictor.py does not call model.predict(X)"
