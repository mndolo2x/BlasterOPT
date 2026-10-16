"""Loader for the standalone Digital Twin HTML scene."""
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "scene_template.html"


def load_scene_html() -> str:
    """Return the raw HTML of the Digital Twin scene template."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Scene template missing at {TEMPLATE_PATH}. "
            "Expected src/ui/digital_twin/scene_template.html."
        )
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def get_scene_template_size() -> int:
    """Return the byte size of the template file."""
    return TEMPLATE_PATH.stat().st_size
