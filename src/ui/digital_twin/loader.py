"""Loader module for injecting scene data into Digital Twin 3D HTML template."""
import json
from pathlib import Path
from typing import Optional, Dict, Any


def load_and_inject_scene(data: Dict[str, Any]) -> Optional[str]:
    """Load scene_template.html and inject window.BLASTOPT_SCENE_DATA JSON string."""
    template_path = Path(__file__).parent / "scene_template.html"
    if not template_path.exists():
        template_path = Path("src/ui/digital_twin/scene_template.html")
    if not template_path.exists():
        return None

    html = template_path.read_text(encoding="utf-8")
    json_data = json.dumps(data)

    # Inject the data right before </head> or at top of file
    injection = f"<script>\nwindow.BLASTOPT_SCENE_DATA = {json_data};\n</script>\n"

    if "</head>" in html:
        html = html.replace("</head>", injection + "</head>", 1)
    else:
        html = injection + html

    return html
