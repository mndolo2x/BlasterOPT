"""
Bench geometry and topography modeling for BlasterOPT 3D engine.
"""

import math
import numpy as np
from typing import Optional, List, Dict, Any, Tuple


class BenchModel:
    """
    Represents a mine bench with real topography, crest, toe, and face angle.
    """

    def __init__(
        self,
        bench_id: str,
        crest_elevation_m: float,
        toe_elevation_m: float,
        face_angle_deg: float,
        topography_points: Optional[np.ndarray] = None,
        bench_width_m: float = 50.0,
        bench_length_m: float = 100.0,
    ) -> None:
        """
        Initialize BenchModel with geometric parameters.

        Args:
            bench_id: Unique identifier for the bench.
            crest_elevation_m: Bench top crest elevation (m).
            toe_elevation_m: Bench floor toe elevation (m).
            face_angle_deg: Bench face angle in degrees (e.g. 70° to 85°).
            topography_points: Optional (N, 3) numpy array of surface points.
            bench_width_m: Width of the bench polygon across X (m).
            bench_length_m: Length of the bench polygon along Y (m).
        """
        if crest_elevation_m <= toe_elevation_m:
            raise ValueError(
                f"crest_elevation_m ({crest_elevation_m}) must be greater than toe_elevation_m ({toe_elevation_m})"
            )
        if not (10.0 <= face_angle_deg <= 90.0):
            raise ValueError(
                f"face_angle_deg ({face_angle_deg}) must be between 10 and 90 degrees"
            )

        self.bench_id = bench_id
        self.crest_elevation_m = crest_elevation_m
        self.toe_elevation_m = toe_elevation_m
        self.face_angle_deg = face_angle_deg
        self.topography_points = topography_points
        self.bench_width_m = bench_width_m
        self.bench_length_m = bench_length_m

    def get_bench_height(self) -> float:
        """
        Returns crest_elevation - toe_elevation.

        Formula:
            bench_height = crest_elevation_m - toe_elevation_m
        """
        # Formula: bench_height = crest_elevation_m - toe_elevation_m
        return float(self.crest_elevation_m - self.toe_elevation_m)

    def get_crest_line(self) -> np.ndarray:
        """
        Returns 3D coordinates of the crest edge along Y.
        """
        # Crest edge at x = 0, z = crest_elevation_m
        y_pts = np.linspace(0.0, self.bench_length_m, 50)
        x_pts = np.zeros_like(y_pts)
        z_pts = np.full_like(y_pts, self.crest_elevation_m)
        return np.column_stack([x_pts, y_pts, z_pts])

    def get_toe_line(self) -> np.ndarray:
        """
        Returns 3D coordinates of the toe edge along Y.

        Bench face geometry formula:
            horizontal_offset = bench_height / tan(face_angle_radians)
        """
        height = self.get_bench_height()
        face_rad = math.radians(self.face_angle_deg)
        # Formula: horizontal_offset = bench_height / tan(face_angle_radians)
        horizontal_offset = height / math.tan(face_rad)

        y_pts = np.linspace(0.0, self.bench_length_m, 50)
        x_pts = np.full_like(y_pts, -horizontal_offset)
        z_pts = np.full_like(y_pts, self.toe_elevation_m)
        return np.column_stack([x_pts, y_pts, z_pts])

    def get_face_surface(self, resolution: int = 50) -> np.ndarray:
        """
        Returns a 3D grid mesh of the bench face.
        Uses the face angle to interpolate between crest and toe.
        If topography_points provided, blends them with the ideal plane.

        Formula:
            face_x(z) = crest_x + (toe_x - crest_x) * (crest_z - z) / bench_height
        """
        height = self.get_bench_height()
        crest_line = self.get_crest_line()
        toe_line = self.get_toe_line()

        z_vals = np.linspace(self.toe_elevation_m, self.crest_elevation_m, resolution)
        y_vals = np.linspace(0.0, self.bench_length_m, resolution)

        Z_grid, Y_grid = np.meshgrid(z_vals, y_vals)

        # Formula: face_x(z) = crest_x + (toe_x - crest_x) * (crest_z - z) / bench_height
        toe_x = toe_line[0, 0]
        crest_x = crest_line[0, 0]
        X_grid = crest_x + (toe_x - crest_x) * (self.crest_elevation_m - Z_grid) / height

        if self.topography_points is not None and len(self.topography_points) > 0:
            # Blend topography perturbation onto X_grid
            mean_topo_noise = np.mean(self.topography_points[:, 0]) * 0.05
            X_grid += mean_topo_noise

        pts = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])
        return pts

    def render(self, plotter: Any = None) -> Dict[str, Any]:
        """
        Adds bench geometry to a PyVista or Plotly plotter/dictionary.
        """
        crest = self.get_crest_line()
        toe = self.get_toe_line()
        face = self.get_face_surface()

        if plotter is not None and hasattr(plotter, "add_mesh"):
            import pyvista as pv
            cloud = pv.PolyData(face)
            plotter.add_mesh(cloud, color="brown", opacity=0.6, label="Bench Face")

        return {
            "bench_id": self.bench_id,
            "crest_points": crest,
            "toe_points": toe,
            "face_surface_points": face,
        }
