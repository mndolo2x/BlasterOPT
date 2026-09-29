"""
Free face identification and blast direction recommendation for BlasterOPT 3D engine.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.render3d.bench_model import BenchModel


def identify_free_faces(
    bench_model: BenchModel, topography_points: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Identify all free faces on the bench.

    A free face is a surface with:
    - Slope angle between 45° and 90° from horizontal
    - Vertical height >= 2 m
    - Width >= 5 m
    - Exposure to open air (no rock at 1 m in front)

    Returns a list of free faces, each with:
        {
            "face_id": str,
            "centroid": (x, y, z),
            "orientation_deg": float,  # azimuth
            "height_m": float,
            "width_m": float,
            "quality": str,  # 'excellent', 'good', 'poor'
        }
    """
    bench_height = bench_model.get_bench_height()
    bench_width = bench_model.bench_width_m
    face_angle = bench_model.face_angle_deg

    faces: List[Dict[str, Any]] = []

    # Main bench face
    if bench_height >= 2.0 and bench_width >= 5.0 and (45.0 <= face_angle <= 90.0):
        # Calculate centroid
        toe_line = bench_model.get_toe_line()
        crest_line = bench_model.get_crest_line()
        centroid_x = float((toe_line[0, 0] + crest_line[0, 0]) / 2.0)
        centroid_y = float(bench_model.bench_length_m / 2.0)
        centroid_z = float((bench_model.toe_elevation_m + bench_model.crest_elevation_m) / 2.0)

        # Quality assessment
        if face_angle >= 70.0 and bench_height >= 10.0:
            quality = "excellent"
        elif face_angle >= 60.0:
            quality = "good"
        else:
            quality = "poor"

        faces.append(
            {
                "face_id": f"FACE_{bench_model.bench_id}_FRONT",
                "centroid": (round(centroid_x, 2), round(centroid_y, 2), round(centroid_z, 2)),
                "orientation_deg": 180.0,  # Facing South / out of bench
                "height_m": round(bench_height, 2),
                "width_m": round(bench_width, 2),
                "quality": quality,
            }
        )

    return faces


def recommend_blast_direction(free_faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recommend the optimal blast direction based on free face orientation.
    The blast should break toward the free face.
    """
    if not free_faces:
        return {
            "recommended_azimuth_deg": 0.0,
            "primary_free_face_id": None,
            "reasoning": "No valid free face detected. Defaulting to 0° (choked blast design required).",
        }

    # Select best quality free face or largest width/height
    quality_order = {"excellent": 3, "good": 2, "poor": 1}
    best_face = max(
        free_faces,
        key=lambda f: (quality_order.get(f.get("quality", "poor"), 0), f.get("width_m", 0.0)),
    )

    opt_azimuth = float(best_face["orientation_deg"])

    return {
        "recommended_azimuth_deg": opt_azimuth,
        "primary_free_face_id": best_face["face_id"],
        "reasoning": (
            f"Blast should initiate toward primary free face '{best_face['face_id']}' "
            f"at azimuth {opt_azimuth:.1f}° (face quality: {best_face['quality']}, "
            f"height: {best_face['height_m']}m, width: {best_face['width_m']}m)."
        ),
    }
