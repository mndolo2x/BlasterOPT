"""
3D rendering and physical analysis package for BlasterOPT / BlastOpt Botswana.
"""

from src.render3d.bench_model import BenchModel
from src.render3d.hole_pattern import HolePattern
from src.render3d.subdrill_analysis import (
    calculate_optimal_subdrill,
    analyze_subdrill_existing,
)
from src.render3d.toe_burden import (
    calculate_toe_burden,
    calculate_relief_holes,
)
from src.render3d.backbreak import (
    predict_backbreak,
    recommend_backbreak_prevention,
)
from src.render3d.free_face import (
    identify_free_faces,
    recommend_blast_direction,
)
from src.render3d.collision import (
    check_hole_collisions,
)
from src.render3d.timing_viz import (
    create_timing_animation,
    add_timing_to_plotter,
)
from src.render3d.renderer import (
    BlastRenderer3D,
    Blast3DRenderer,
)

__all__ = [
    "BenchModel",
    "HolePattern",
    "calculate_optimal_subdrill",
    "analyze_subdrill_existing",
    "calculate_toe_burden",
    "calculate_relief_holes",
    "predict_backbreak",
    "recommend_backbreak_prevention",
    "identify_free_faces",
    "recommend_blast_direction",
    "check_hole_collisions",
    "create_timing_animation",
    "add_timing_to_plotter",
    "BlastRenderer3D",
    "Blast3DRenderer",
]
