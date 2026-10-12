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


def build_bench_mesh(bench: BenchGeometry, subdrill_m: float = 1.5, resolution: int = 30) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build an irregular 3D topographic surface mesh for the bench and pit floor.

    Generates realistic terrain elevation variation, crest surface topography,
    vertical bench slope, and pit floor.

    Returns a dict with:
        crest: {x, y, z} meshgrid arrays
        face: {x, y, z} meshgrid arrays
        floor: {x, y, z} meshgrid arrays
        bottom_z: bottom elevation matching subdrilled hole toes
        get_collar_z: helper lookup function for (x, y) coordinates
    """
    height = compute_bench_height(bench)
    face_offset = compute_face_offset(bench)
    bottom_z = bench.toe_elevation_m - max(subdrill_m, 0.0)

    # Top surface (crest plane) with realistic micro-topography
    x_crest = np.linspace(0, bench.length_m, resolution)
    y_crest = np.linspace(0, bench.width_m, resolution)
    X_crest, Y_crest = np.meshgrid(x_crest, y_crest)

    # Natural topographic variation (waves + fine noise)
    np.random.seed(42)
    topo_noise = (
        0.25 * np.sin(X_crest / 6.0) * np.cos(Y_crest / 5.0) +
        0.15 * np.cos(X_crest / 3.0) +
        np.random.uniform(-0.08, 0.08, size=X_crest.shape)
    )
    Z_crest = bench.crest_elevation_m + topo_noise

    # Face surface (slope from crest to toe)
    x_face = np.linspace(0, bench.length_m, resolution)
    z_face = np.linspace(bench.crest_elevation_m, bottom_z, resolution)
    X_face, Z_face = np.meshgrid(x_face, z_face)
    Y_face = np.full_like(X_face, 0.0)

    # Pit Floor
    y_floor = np.linspace(-bench.width_m * 0.3, 0.0, resolution)
    X_floor, Y_floor = np.meshgrid(x_crest, y_floor)
    Z_floor = np.full_like(X_floor, bottom_z) + np.random.uniform(-0.05, 0.05, size=X_floor.shape)

    def get_collar_z(x: float, y: float) -> float:
        """Calculate local collar elevation on topographic mesh."""
        noise = (
            0.25 * np.sin(x / 6.0) * np.cos(y / 5.0) +
            0.15 * np.cos(x / 3.0)
        )
        return float(bench.crest_elevation_m + noise)

    return {
        "crest": {"x": X_crest, "y": Y_crest, "z": Z_crest},
        "face": {"x": X_face, "y": Y_face, "z": Z_face},
        "floor": {"x": X_floor, "y": Y_floor, "z": Z_floor},
        "bottom_z": bottom_z,
        "get_collar_z": get_collar_z,
    }
