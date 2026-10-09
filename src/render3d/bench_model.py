"""
3D bench geometry: crest, toe, face angle, free face, topography.
"""
import numpy as np
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class BenchGeometry(BaseModel):
    """Geometric definition of a mine bench."""
    bench_id: str = Field(..., description="Bench identifier, e.g., 'B14'")
    crest_elevation_m: float = Field(..., description="Top of bench (m)")
    toe_elevation_m: float = Field(..., description="Bottom of bench (m)")
    face_angle_deg: float = Field(75.0, description="Face angle from horizontal")
    length_m: float = Field(100.0, description="Bench length along strike")
    width_m: float = Field(60.0, description="Bench width across strike")
    free_face_direction: str = Field("north", description="Direction of free face")


def compute_bench_height(bench: BenchGeometry) -> float:
    """Return crest elevation minus toe elevation."""
    height = bench.crest_elevation_m - bench.toe_elevation_m
    if height <= 0:
        raise ValueError(f"Bench height must be positive, got {height}")
    return height


def compute_face_offset(bench: BenchGeometry) -> float:
    """
    Horizontal offset from crest to toe due to face angle.

    Formula:
        offset = height / tan(face_angle)
    """
    height = compute_bench_height(bench)
    angle_rad = np.radians(bench.face_angle_deg)
    return float(height / np.tan(angle_rad))


def build_bench_mesh(bench: BenchGeometry, resolution: int = 20) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build a 3D surface mesh for the bench.

    Returns a dict with:
        crest: {x, y, z} meshgrid arrays
        face: {x, y, z} meshgrid arrays
    """
    height = compute_bench_height(bench)
    face_offset = compute_face_offset(bench)

    # Top surface (crest plane)
    x_crest = np.linspace(0, bench.length_m, resolution)
    y_crest = np.linspace(0, bench.width_m, resolution)
    X_crest, Y_crest = np.meshgrid(x_crest, y_crest)
    Z_crest = np.full_like(X_crest, bench.crest_elevation_m)

    # Face surface (slope from crest to toe)
    x_face = np.linspace(0, bench.length_m, resolution)
    z_face = np.linspace(bench.crest_elevation_m, bench.toe_elevation_m, resolution)
    X_face, Z_face = np.meshgrid(x_face, z_face)
    Y_face = np.full_like(X_face, 0.0)  # Free face at Y=0

    return {
        "crest": {"x": X_crest, "y": Y_crest, "z": Z_crest},
        "face": {"x": X_face, "y": Y_face, "z": Z_face},
    }
