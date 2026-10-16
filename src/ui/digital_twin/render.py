"""Loader for the standalone Digital Twin HTML scene with CSS overrides."""
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "scene_template.html"

# CSS to hide the HTML dashboard's own chrome when embedded
EMBED_CSS = """
<style id="blastopt-embed-overrides">
/* Hide the top header bar */
header { display: none !important; }

/* Hide the left scene controls panel */
#left { display: none !important; }

/* Hide the right info panels */
#right { display: none !important; }

/* Hide the footer */
footer { display: none !important; }

/* Reflow the grid to a single column with the viewport only */
#app {
    grid-template-columns: 1fr !important;
    grid-template-rows: 1fr !important;
    height: 100vh !important;
}

/* Make sure the canvas fills the frame */
#viewport {
    grid-column: 1 !important;
    grid-row: 1 !important;
    min-height: 500px !important;
}

/* Hide the corner legend, compass overlay, tooltips */
.cornerLegend, .compass, .faultLabel, .tooltip {
    display: none !important;
}

/* Hide the viewport top toolbar (view buttons) */
.viewportTop .sceneTag { display: none !important; }
</style>
"""


def load_scene_html() -> str:
    """Return the Digital Twin scene HTML with embedded CSS overrides."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Scene template missing at {TEMPLATE_PATH}."
        )
    raw = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Inject the override CSS right before </head>
    if "</head>" in raw:
        raw = raw.replace("</head>", EMBED_CSS + "</head>", 1)
    return raw


def get_scene_template_size() -> int:
    """Return the byte size of the raw template file."""
    return TEMPLATE_PATH.stat().st_size
