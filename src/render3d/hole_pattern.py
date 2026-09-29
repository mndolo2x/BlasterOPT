"""
Drill hole pattern placement and coordinate generation for BlasterOPT 3D engine.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.render3d.bench_model import BenchModel


class Hole:
    """Represents a single drill hole in 3D space."""

    def __init__(
        self,
        hole_id: str,
        x: float,
        y: float,
        z: float,
        depth_m: float,
        diameter_mm: float = 250.0,
        dip_deg: float = 90.0,
        azimuth_deg: float = 0.0,
    ) -> None:
        self.hole_id = hole_id
        self.x = x
        self.y = y
        self.z = z
        self.depth_m = depth_m
        self.diameter_mm = diameter_mm
        self.dip_deg = dip_deg
        self.azimuth_deg = azimuth_deg

    def get_toe_coordinate(self) -> Tuple[float, float, float]:
        """
        Calculate 3D toe coordinate based on depth, dip, and azimuth.

        Formulas:
            horizontal_drill = depth * cos(dip_rad)
            toe_x = x + horizontal_drill * cos(azimuth_rad)
            toe_y = y + horizontal_drill * sin(azimuth_rad)
            toe_z = z - depth * sin(dip_rad)
        """
        dip_rad = math.radians(self.dip_deg)
        az_rad = math.radians(self.azimuth_deg)
        h_length = self.depth_m * math.cos(dip_rad)
        toe_x = self.x + h_length * math.cos(az_rad)
        toe_y = self.y + h_length * math.sin(az_rad)
        toe_z = self.z - self.depth_m * math.sin(dip_rad)
        return (toe_x, toe_y, toe_z)


class HolePattern:
    """
    Represents a drill pattern with hole positions, depths, and angles.
    """

    def __init__(
        self,
        burden_m: float,
        spacing_m: float,
        bench: BenchModel,
        pattern_type: str = "staggered",
        hole_diameter_mm: float = 250.0,
        subdrill_m: float = 1.5,
    ) -> None:
        """
        Initialize HolePattern.

        Args:
            burden_m: Burden distance between rows (m).
            spacing_m: Spacing distance between holes in a row (m).
            bench: BenchModel instance providing bench geometry.
            pattern_type: 'square', 'staggered', 'echelon', or 'v_pattern'.
            hole_diameter_mm: Hole diameter (mm).
            subdrill_m: Subdrill depth below bench toe elevation (m).
        """
        valid_types = ["square", "staggered", "echelon", "v_pattern"]
        if pattern_type not in valid_types:
            raise ValueError(
                f"Invalid pattern_type '{pattern_type}'. Must be one of {valid_types}"
            )

        self.burden_m = burden_m
        self.spacing_m = spacing_m
        self.bench = bench
        self.pattern_type = pattern_type
        self.hole_diameter_mm = hole_diameter_mm
        self.subdrill_m = subdrill_m

    def generate_holes(self, n_rows: int, n_per_row: int) -> List[Hole]:
        """
        Generates hole coordinates based on pattern_type.

        Formulas:
            Square pattern:
                x = col * spacing, y = row * burden
            Staggered pattern:
                x = col * spacing + (row % 2) * spacing / 2, y = row * burden
            Echelon pattern:
                x = col * spacing + row * spacing * 0.25, y = row * burden
            V-pattern:
                x = col * spacing + abs(row - n_rows / 2) * spacing * 0.3, y = row * burden
        """
        holes: List[Hole] = []
        bench_height = self.bench.get_bench_height()
        depth = bench_height + self.subdrill_m

        for row in range(n_rows):
            y = float(row * self.burden_m)
            for col in range(n_per_row):
                if self.pattern_type == "square":
                    # Formula: x = col * spacing
                    x = float(col * self.spacing_m)
                elif self.pattern_type == "staggered":
                    # Formula: x = col * spacing + (row % 2) * spacing / 2
                    x = float(col * self.spacing_m + (row % 2) * (self.spacing_m / 2.0))
                elif self.pattern_type == "echelon":
                    # Formula: x = col * spacing + row * spacing * 0.25
                    x = float(col * self.spacing_m + row * self.spacing_m * 0.25)
                elif self.pattern_type == "v_pattern":
                    # Formula: x = col * spacing + abs(row - n_rows / 2) * spacing * 0.3
                    x = float(
                        col * self.spacing_m
                        + abs(row - n_rows / 2.0) * self.spacing_m * 0.3
                    )
                else:
                    x = float(col * self.spacing_m)

                z = float(self.bench.crest_elevation_m)
                hole_id = f"H_{row+1}_{col+1}"
                hole = Hole(
                    hole_id=hole_id,
                    x=x,
                    y=y,
                    z=z,
                    depth_m=depth,
                    diameter_mm=self.hole_diameter_mm,
                )
                holes.append(hole)

        return holes

    def render(self, plotter: Any = None) -> List[Dict[str, Any]]:
        """
        Adds hole markers to a PyVista plotter or returns list of hole dicts.
        """
        holes = self.generate_holes(3, 4)
        out = []
        for h in holes:
            toe_x, toe_y, toe_z = h.get_toe_coordinate()
            out.append(
                {
                    "hole_id": h.hole_id,
                    "collar": (h.x, h.y, h.z),
                    "toe": (toe_x, toe_y, toe_z),
                }
            )
        return out
