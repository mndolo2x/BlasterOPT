"""
3D Blast Pattern Design Module for BlastOpt Botswana.
Provides comprehensive 3D blast pattern design, visualization, and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of blast pattern layouts."""
    RECTANGULAR = "rectangular"
    STAGGERED = "staggered"
    ECHELON = "echelon"
    V_PATTERN = "v_pattern"
    CUSTOM = "custom"


@dataclass
class BlastHole:
    """Represents a single blast hole in 3D space."""
    hole_id: str
    x: float  # East coordinate (m)
    y: float  # North coordinate (m)
    z: float  # Elevation coordinate (m)
    collar_elevation: float
    toe_elevation: float
    hole_length: float
    hole_diameter_mm: float
    burden_m: float
    spacing_m: float
    stemming_m: float
    powder_factor_kg_m3: float
    charge_mass_kg: float
    deck_info: Optional[Dict[str, Any]] = None
    deviation_x: float = 0.0
    deviation_y: float = 0.0
    deviation_z: float = 0.0


@dataclass
class BenchGeometry:
    """Represents bench geometry for 3D modeling."""
    bench_height_m: float
    bench_width_m: float
    bench_slope_deg: float
    crest_elevation: float
    toe_elevation: float
    face_angle_deg: float
    crest_to_toe_distance: float
    bench_length_m: float
    azimuth_deg: float


@dataclass
class BlastPattern3D:
    """Represents a complete 3D blast pattern."""
    pattern_id: str
    pattern_type: PatternType
    bench_geometry: BenchGeometry
    holes: List[BlastHole]
    initiation_sequence: List[Dict[str, Any]]
    timing_ms: List[int]
    design_parameters: Dict[str, Any]
    created_at: str
    version: int = 1


class PatternGenerator3D:
    """Generator for 3D blast patterns."""

    def __init__(self):
        """Initialize the 3D pattern generator."""
        self.hole_counter = 0

    def generate_rectangular_pattern(
        self,
        bench_geometry: BenchGeometry,
        burden_m: float,
        spacing_m: float,
        hole_diameter_mm: float,
        stemming_m: float,
        powder_factor_kg_m3: float,
        num_rows: int = 5,
        num_cols: int = 6,
        start_x: float = 0.0,
        start_y: float = 0.0
    ) -> List[BlastHole]:
        """
        Generate rectangular 3D blast pattern.

        Args:
            bench_geometry: Bench geometry parameters
            burden_m: Burden distance (m)
            spacing_m: Spacing distance (m)
            hole_diameter_mm: Hole diameter (mm)
            stemming_m: Stemming length (m)
            powder_factor_kg_m3: Powder factor (kg/m3)
            num_rows: Number of rows
            num_cols: Number of columns
            start_x: Starting X coordinate
            start_y: Starting Y coordinate

        Returns:
            List of BlastHole objects
        """
        holes = []

        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * spacing_m
                y = start_y + row * burden_m
                z = bench_geometry.crest_elevation

                hole = self._create_hole(
                    x=x, y=y, z=z,
                    bench_geometry=bench_geometry,
                    burden_m=burden_m,
                    spacing_m=spacing_m,
                    hole_diameter_mm=hole_diameter_mm,
                    stemming_m=stemming_m,
                    powder_factor_kg_m3=powder_factor_kg_m3
                )
                holes.append(hole)

        return holes

    def generate_staggered_pattern(
        self,
        bench_geometry: BenchGeometry,
        burden_m: float,
        spacing_m: float,
        hole_diameter_mm: float,
        stemming_m: float,
        powder_factor_kg_m3: float,
        num_rows: int = 5,
        num_cols: int = 6,
        start_x: float = 0.0,
        start_y: float = 0.0
    ) -> List[BlastHole]:
        """
        Generate staggered 3D blast pattern.

        Args:
            bench_geometry: Bench geometry parameters
            burden_m: Burden distance (m)
            spacing_m: Spacing distance (m)
            hole_diameter_mm: Hole diameter (mm)
            stemming_m: Stemming length (m)
            powder_factor_kg_m3: Powder factor (kg/m3)
            num_rows: Number of rows
            num_cols: Number of columns
            start_x: Starting X coordinate
            start_y: Starting Y coordinate

        Returns:
            List of BlastHole objects
        """
        holes = []

        for row in range(num_rows):
            for col in range(num_cols):
                # Stagger every other row
                offset = (spacing_m / 2) if row % 2 == 1 else 0

                x = start_x + col * spacing_m + offset
                y = start_y + row * burden_m
                z = bench_geometry.crest_elevation

                hole = self._create_hole(
                    x=x, y=y, z=z,
                    bench_geometry=bench_geometry,
                    burden_m=burden_m,
                    spacing_m=spacing_m,
                    hole_diameter_mm=hole_diameter_mm,
                    stemming_m=stemming_m,
                    powder_factor_kg_m3=powder_factor_kg_m3
                )
                holes.append(hole)

        return holes

    def generate_echelon_pattern(
        self,
        bench_geometry: BenchGeometry,
        burden_m: float,
        spacing_m: float,
        hole_diameter_mm: float,
        stemming_m: float,
        powder_factor_kg_m3: float,
        num_rows: int = 5,
        num_cols: int = 6,
        start_x: float = 0.0,
        start_y: float = 0.0,
        echelon_offset_m: float = 2.0
    ) -> List[BlastHole]:
        """
        Generate echelon 3D blast pattern.

        Args:
            bench_geometry: Bench geometry parameters
            burden_m: Burden distance (m)
            spacing_m: Spacing distance (m)
            hole_diameter_mm: Hole diameter (mm)
            stemming_m: Stemming length (m)
            powder_factor_kg_m3: Powder factor (kg/m3)
            num_rows: Number of rows
            num_cols: Number of columns
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            echelon_offset_m: Echelon offset (m)

        Returns:
            List of BlastHole objects
        """
        holes = []

        for row in range(num_rows):
            for col in range(num_cols):
                # Echelon offset increases with each row
                offset = row * echelon_offset_m

                x = start_x + col * spacing_m + offset
                y = start_y + row * burden_m
                z = bench_geometry.crest_elevation

                hole = self._create_hole(
                    x=x, y=y, z=z,
                    bench_geometry=bench_geometry,
                    burden_m=burden_m,
                    spacing_m=spacing_m,
                    hole_diameter_mm=hole_diameter_mm,
                    stemming_m=stemming_m,
                    powder_factor_kg_m3=powder_factor_kg_m3
                )
                holes.append(hole)

        return holes

    def generate_v_pattern(
        self,
        bench_geometry: BenchGeometry,
        burden_m: float,
        spacing_m: float,
        hole_diameter_mm: float,
        stemming_m: float,
        powder_factor_kg_m3: float,
        num_rows: int = 5,
        num_cols: int = 6,
        start_x: float = 0.0,
        start_y: float = 0.0,
        v_angle_deg: float = 30.0
    ) -> List[BlastHole]:
        """
        Generate V-shaped 3D blast pattern.

        Args:
            bench_geometry: Bench geometry parameters
            burden_m: Burden distance (m)
            spacing_m: Spacing distance (m)
            hole_diameter_mm: Hole diameter (mm)
            stemming_m: Stemming length (m)
            powder_factor_kg_m3: Powder factor (kg/m3)
            num_rows: Number of rows
            num_cols: Number of columns
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            v_angle_deg: V-pattern angle (degrees)

        Returns:
            List of BlastHole objects
        """
        holes = []
        v_angle_rad = np.radians(v_angle_deg)

        center_col = num_cols / 2

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate V-pattern offset
                col_offset = col - center_col
                v_offset = col_offset * np.tan(v_angle_rad) * burden_m

                x = start_x + col * spacing_m
                y = start_y + row * burden_m + v_offset
                z = bench_geometry.crest_elevation

                hole = self._create_hole(
                    x=x, y=y, z=z,
                    bench_geometry=bench_geometry,
                    burden_m=burden_m,
                    spacing_m=spacing_m,
                    hole_diameter_mm=hole_diameter_mm,
                    stemming_m=stemming_m,
                    powder_factor_kg_m3=powder_factor_kg_m3
                )
                holes.append(hole)

        return holes

    def _create_hole(
        self,
        x: float,
        y: float,
        z: float,
        bench_geometry: BenchGeometry,
        burden_m: float,
        spacing_m: float,
        hole_diameter_mm: float,
        stemming_m: float,
        powder_factor_kg_m3: float
    ) -> BlastHole:
        """
        Create a single blast hole with calculations.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            bench_geometry: Bench geometry
            burden_m: Burden distance
            spacing_m: Spacing distance
            hole_diameter_mm: Hole diameter
            stemming_m: Stemming length
            powder_factor_kg_m3: Powder factor

        Returns:
            BlastHole object
        """
        self.hole_counter += 1
        hole_id = f"H{self.hole_counter:03d}"

        # Calculate hole length based on bench geometry
        hole_length = bench_geometry.bench_height_m + 1.0  # Add subdrilling

        # Calculate toe elevation
        toe_elevation = z - hole_length

        # Calculate charge mass
        charge_volume = (np.pi * (hole_diameter_mm / 2000) ** 2) * (hole_length - stemming_m)
        charge_mass_kg = charge_volume * powder_factor_kg_m3 * 1000  # Convert to kg

        return BlastHole(
            hole_id=hole_id,
            x=x,
            y=y,
            z=z,
            collar_elevation=z,
            toe_elevation=toe_elevation,
            hole_length=hole_length,
            hole_diameter_mm=hole_diameter_mm,
            burden_m=burden_m,
            spacing_m=spacing_m,
            stemming_m=stemming_m,
            powder_factor_kg_m3=powder_factor_kg_m3,
            charge_mass_kg=charge_mass_kg
        )


class HoleDeviationCompensator:
    """Compensates for hole deviation in 3D pattern design."""

    def __init__(self, max_deviation_deg: float = 3.0):
        """
        Initialize deviation compensator.

        Args:
            max_deviation_deg: Maximum expected deviation (degrees)
        """
        self.max_deviation_deg = max_deviation_deg

    def add_deviation_to_holes(
        self,
        holes: List[BlastHole],
        deviation_profile: str = "random"
    ) -> List[BlastHole]:
        """
        Add deviation to holes for realistic modeling.

        Args:
            holes: List of blast holes
            deviation_profile: Type of deviation profile

        Returns:
            List of holes with deviation added
        """
        compensated_holes = []

        for hole in holes:
            if deviation_profile == "random":
                deviation_x = np.random.uniform(-self.max_deviation_deg, self.max_deviation_deg)
                deviation_y = np.random.uniform(-self.max_deviation_deg, self.max_deviation_deg)
                deviation_z = np.random.uniform(-self.max_deviation_deg, self.max_deviation_deg)
            elif deviation_profile == "systematic":
                # Systematic deviation based on hole length
                deviation_factor = hole.hole_length / 20.0
                deviation_x = deviation_factor * 1.0
                deviation_y = deviation_factor * 1.0
                deviation_z = deviation_factor * 0.5
            else:
                deviation_x = 0.0
                deviation_y = 0.0
                deviation_z = 0.0

            # Convert to radians
            dev_x_rad = np.radians(deviation_x)
            dev_y_rad = np.radians(deviation_y)
            dev_z_rad = np.radians(deviation_z)

            compensated_hole = BlastHole(
                hole_id=hole.hole_id,
                x=hole.x,
                y=hole.y,
                z=hole.z,
                collar_elevation=hole.collar_elevation,
                toe_elevation=hole.toe_elevation,
                hole_length=hole.hole_length,
                hole_diameter_mm=hole.hole_diameter_mm,
                burden_m=hole.burden_m,
                spacing_m=hole.spacing_m,
                stemming_m=hole.stemming_m,
                powder_factor_kg_m3=hole.powder_factor_kg_m3,
                charge_mass_kg=hole.charge_mass_kg,
                deviation_x=dev_x_rad,
                deviation_y=dev_y_rad,
                deviation_z=dev_z_rad
            )
            compensated_holes.append(compensated_hole)

        return compensated_holes

    def calculate_actual_toe_position(
        self,
        hole: BlastHole
    ) -> Tuple[float, float, float]:
        """
        Calculate actual toe position considering deviation.

        Args:
            hole: Blast hole with deviation

        Returns:
            Tuple of (x, y, z) coordinates of actual toe position
        """
        # Calculate deviation effects
        x_deviation = hole.hole_length * np.tan(hole.deviation_x)
        y_deviation = hole.hole_length * np.tan(hole.deviation_y)
        z_deviation = hole.hole_length * np.tan(hole.deviation_z)

        actual_toe_x = hole.x + x_deviation
        actual_toe_y = hole.y + y_deviation
        actual_toe_z = hole.toe_elevation + z_deviation

        return (actual_toe_x, actual_toe_y, actual_toe_z)


class DeckingDesigner:
    """Designer for multi-deck blasting configurations."""

    def design_decking(
        self,
        hole: BlastHole,
        num_decks: int = 2,
        deck_ratios: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Design decking configuration for a hole.

        Args:
            hole: Blast hole to design decking for
            num_decks: Number of decks
            deck_ratios: Custom deck ratios (optional)

        Returns:
            Dictionary with decking configuration
        """
        if deck_ratios is None:
            # Equal distribution by default
            deck_ratios = [1.0 / num_decks] * num_decks

        # Normalize ratios
        total_ratio = sum(deck_ratios)
        deck_ratios = [r / total_ratio for r in deck_ratios]

        # Calculate deck lengths
        chargeable_length = hole.hole_length - hole.stemming_m
        deck_lengths = [chargeable_length * ratio for ratio in deck_ratios]

        # Calculate deck positions
        deck_positions = []
        current_position = hole.stemming_m

        for i, deck_length in enumerate(deck_lengths):
            deck_start = current_position
            deck_end = current_position + deck_length
            deck_positions.append({
                'deck_number': i + 1,
                'start_depth_m': deck_start,
                'end_depth_m': deck_end,
                'length_m': deck_length,
                'charge_mass_kg': hole.charge_mass_kg * deck_ratios[i]
            })
            current_position = deck_end

        return {
            'hole_id': hole.hole_id,
            'num_decks': num_decks,
            'deck_ratios': deck_ratios,
            'deck_positions': deck_positions,
            'total_charge_mass_kg': hole.charge_mass_kg
        }


def create_bench_geometry(
    bench_height_m: float = 12.0,
    bench_width_m: float = 20.0,
    bench_slope_deg: float = 75.0,
    crest_elevation: float = 100.0,
    face_angle_deg: float = 70.0,
    bench_length_m: float = 100.0,
    azimuth_deg: float = 0.0,
    topography_data: Optional[List[Tuple[float, float, float]]] = None
) -> BenchGeometry:
    """
    Create bench geometry object.

    Args:
        bench_height_m: Bench height (m)
        bench_width_m: Bench width (m)
        bench_slope_deg: Bench slope angle (degrees)
        crest_elevation: Crest elevation (m)
        face_angle_deg: Face angle (degrees)
        bench_length_m: Bench length (m)
        azimuth_deg: Bench azimuth (degrees)
        topography_data: Optional list of (x, y, z) points for custom topography

    Returns:
        BenchGeometry object
    """
    toe_elevation = crest_elevation - bench_height_m
    crest_to_toe_distance = bench_height_m / np.tan(np.radians(face_angle_deg))

    return BenchGeometry(
        bench_height_m=bench_height_m,
        bench_width_m=bench_width_m,
        bench_slope_deg=bench_slope_deg,
        crest_elevation=crest_elevation,
        toe_elevation=toe_elevation,
        face_angle_deg=face_angle_deg,
        crest_to_toe_distance=crest_to_toe_distance,
        bench_length_m=bench_length_m,
        azimuth_deg=azimuth_deg
    )


class SubdrillingOptimizer:
    """Optimizer for subdrilling depth and floor breakage analysis in 3D blast patterns."""

    def __init__(self):
        """Initialize subdrilling optimizer."""
        pass

    def calculate_optimal_subdrilling(
        self,
        bench_height_m: float,
        hole_diameter_mm: float,
        rock_type: str = "medium",
        burden_m: Optional[float] = None,
        ucs_mpa: Optional[float] = None,
        bedding_dip_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate optimal subdrilling depth based on rock properties, hole diameter, and bench geometry.

        Args:
            bench_height_m: Bench height (m)
            hole_diameter_mm: Hole diameter (mm)
            rock_type: Rock type ('soft', 'medium', 'hard', 'very_hard')
            burden_m: Optional design burden (m)
            ucs_mpa: Optional Uniaxial Compressive Strength (MPa)
            bedding_dip_deg: Dip angle of rock bedding plane relative to pit floor (deg)

        Returns:
            Dictionary with optimal subdrilling parameters, cost impact, and risk classification.
        """
        # Base rule of thumb: 8-12x hole diameter or 0.2-0.3x burden
        base_subdrilling = (hole_diameter_mm / 1000.0) * 10.0
        if burden_m is not None and burden_m > 0:
            burden_subdrilling = burden_m * 0.25
            base_subdrilling = 0.5 * (base_subdrilling + burden_subdrilling)

        # Adjust for rock hardness / UCS
        rock_factor = {
            "soft": 0.75,
            "medium": 1.0,
            "hard": 1.25,
            "very_hard": 1.45,
        }.get(rock_type.lower(), 1.0)

        if ucs_mpa is not None:
            if ucs_mpa < 50:
                rock_factor = 0.75
            elif ucs_mpa < 120:
                rock_factor = 1.0
            elif ucs_mpa < 200:
                rock_factor = 1.25
            else:
                rock_factor = 1.45

        # Adjust for structural bedding dip: dipping into pit requires more subdrill to shear toe
        dip_factor = 1.0
        if bedding_dip_deg > 10.0:
            dip_factor = 1.15
        elif bedding_dip_deg < -10.0:
            dip_factor = 0.90

        # Height factor (higher benches require slightly deeper subdrill to maintain floor grade)
        height_factor = 1.0 if bench_height_m < 15.0 else 1.10

        optimal_subdrilling = base_subdrilling * rock_factor * dip_factor * height_factor
        min_subdrilling = optimal_subdrilling * 0.80
        max_subdrilling = optimal_subdrilling * 1.25

        subdrill_pct = (optimal_subdrilling / max(bench_height_m, 1.0)) * 100.0

        # Risk assessment
        if subdrill_pct < 6.0:
            risk = "High risk of toe humps / hard bottom requiring secondary blasting"
            toe_breakage_prob = 0.72
        elif subdrill_pct <= 14.0:
            risk = "Optimal grade level breakage; minimal floor damage"
            toe_breakage_prob = 0.96
        else:
            risk = "Excessive subdrilling: elevated ground vibration and grade dilution"
            toe_breakage_prob = 0.99

        return {
            "optimal_subdrilling_m": round(optimal_subdrilling, 2),
            "min_subdrilling_m": round(min_subdrilling, 2),
            "max_subdrilling_m": round(max_subdrilling, 2),
            "percentage_of_bench_height": round(subdrill_pct, 1),
            "rock_factor": rock_factor,
            "dip_factor": dip_factor,
            "toe_breakage_probability": toe_breakage_prob,
            "risk_assessment": risk,
            "recommendation": self._get_subdrilling_recommendation(optimal_subdrilling, bench_height_m),
        }

    def _get_subdrilling_recommendation(self, subdrilling: float, bench_height: float) -> str:
        """Get subdrilling recommendation string."""
        percentage = (subdrilling / max(bench_height, 1.0)) * 100.0
        if percentage < 6.0:
            return "Subdrilling is low — increase by 0.3-0.5m to avoid high toe costs"
        elif percentage <= 12.0:
            return "Subdrilling is in the optimal window for clean floor level grade"
        elif percentage <= 16.0:
            return "Subdrilling is acceptable for hard/massive rock formation"
        else:
            return "Excessive subdrilling — decrease subdrill to reduce drilling costs and vibration"

    def optimize_subdrilling_for_pattern(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        rock_type: str = "medium",
    ) -> Dict[str, Any]:
        """Analyze and optimize subdrilling across all holes in an active pattern."""
        if not holes:
            return {"error": "No holes provided"}

        avg_diam = float(np.mean([h.hole_diameter_mm for h in holes]))
        avg_burden = float(np.mean([h.burden_m for h in holes]))

        opt_info = self.calculate_optimal_subdrilling(
            bench_height_m=bench_geometry.bench_height_m,
            hole_diameter_mm=avg_diam,
            rock_type=rock_type,
            burden_m=avg_burden,
        )

        # Per-hole subdrill depths
        actual_subdrills = [
            max(0.0, bench_geometry.toe_elevation - h.toe_elevation)
            for h in holes
        ]

        avg_actual = float(np.mean(actual_subdrills)) if actual_subdrills else 0.0
        optimal_val = opt_info["optimal_subdrilling_m"]
        variance = avg_actual - optimal_val

        return {
            "target_optimal_subdrilling_m": optimal_val,
            "current_avg_subdrilling_m": round(avg_actual, 2),
            "variance_m": round(variance, 2),
            "opt_info": opt_info,
            "adequacy_score_pct": max(0.0, min(100.0, round(100.0 - abs(variance) * 25.0, 1))),
        }


class ToeBurdenAnalyzer:
    """Analyzer for toe burden and relief calculations in 3D benches."""

    def __init__(self):
        """Initialize toe burden analyzer."""
        pass

    def analyze_toe_burden(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        face_angle_deg: float = 70.0,
        face_data: Optional[List[Tuple[float, float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze toe burden and face confinement for blast pattern.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            face_angle_deg: Face angle (degrees)
            face_data: Optional 3D face survey points

        Returns:
            Dictionary with toe burden analysis, relief factor, and mitigation advice.
        """
        if not holes:
            return {"error": "No holes provided"}

        # Theoretical crest-to-toe offset due to slope
        rad_face = np.radians(face_angle_deg if face_angle_deg > 0 else 70.0)
        face_toe_offset_m = bench_geometry.bench_height_m / np.tan(rad_face)

        # Filter front row holes (holes closest to face Y coordinate)
        min_y = min(h.y for h in holes)
        front_row = [h for h in holes if h.y <= min_y + (h.burden_m * 0.75)]
        if not front_row:
            front_row = holes

        toe_burdens = []
        toe_ratios = []
        for h in front_row:
            # Distance from toe to bench face line at toe elevation
            # Bench face toe is located at y = -face_toe_offset_m if crest is at y=0
            actual_toe_burden = h.y + face_toe_offset_m
            # Account for hole deviation at toe
            actual_toe_burden += h.deviation_y
            toe_burdens.append(actual_toe_burden)

            nominal_b = max(h.burden_m, 1.0)
            toe_ratios.append(actual_toe_burden / nominal_b)

        avg_toe_burden = float(np.mean(toe_burdens))
        min_toe_burden = float(np.min(toe_burdens))
        max_toe_burden = float(np.max(toe_burdens))
        avg_ratio = float(np.mean(toe_ratios))

        toe_issues: List[str] = []
        if avg_ratio > 1.30:
            toe_issues.append("Critically excessive toe burden — high risk of toe boulders and severe flyrock")
        elif avg_ratio > 1.15:
            toe_issues.append("Moderate toe confinement — consider angle drilling or bottom deck intensification")
        elif avg_ratio < 0.70:
            toe_issues.append("Under-burdened toe — danger of violent face blowout and flyrock")

        # Relief calculations: effective relief ratio (Spacing / Toe Burden)
        avg_spacing = float(np.mean([h.spacing_m for h in front_row]))
        relief_ratio = avg_spacing / max(avg_toe_burden, 0.1)

        # Effective expansion volume per hole (m3)
        relief_volume_m3 = avg_toe_burden * avg_spacing * bench_geometry.bench_height_m

        return {
            "theoretical_toe_offset_m": round(face_toe_offset_m, 2),
            "avg_toe_burden_m": round(avg_toe_burden, 2),
            "min_toe_burden_m": round(min_toe_burden, 2),
            "max_toe_burden_m": round(max_toe_burden, 2),
            "toe_to_nominal_ratio": round(avg_ratio, 2),
            "relief_ratio_sb": round(relief_ratio, 2),
            "relief_volume_m3_per_hole": round(relief_volume_m3, 1),
            "toe_issues": toe_issues,
            "status": "Warning" if toe_issues else "Safe",
            "recommendation": self._get_toe_burden_recommendation(avg_ratio, toe_issues),
        }

    def _get_toe_burden_recommendation(self, adequacy: float, issues: List[str]) -> str:
        """Get toe burden recommendation string."""
        if not issues:
            return "Toe burden and relief are well-balanced for optimal fragmentation"
        return "; ".join(issues)

    def calculate_relief_factors(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        inter_row_delay_ms: float = 42.0,
        inter_hole_delay_ms: float = 17.0,
    ) -> Dict[str, Any]:
        """
        Calculate dynamic relief factors based on timing and geometric expansion space.
        """
        if not holes:
            return {"error": "No holes provided"}

        avg_burden = float(np.mean([h.burden_m for h in holes]))
        avg_spacing = float(np.mean([h.spacing_m for h in holes]))

        # Delay relief ratio (ms / m of burden)
        delay_relief_ms_m = inter_row_delay_ms / max(avg_burden, 0.1)

        # Ideal range is 15-30 ms per meter of burden
        if delay_relief_ms_m < 12.0:
            relief_timing_status = "Too fast — choked rock movement, elevated backbreak"
        elif delay_relief_ms_m <= 30.0:
            relief_timing_status = "Optimal delay relief — clean forward muckpile cast"
        else:
            relief_timing_status = "Overly relaxed delay — potential cut-off risk"

        return {
            "delay_relief_ms_per_m_burden": round(delay_relief_ms_m, 2),
            "expansion_relief_ratio": round(avg_spacing / max(avg_burden, 0.1), 2),
            "timing_status": relief_timing_status,
            "swell_volume_pct_expected": 35.0,
        }


class BackbreakPredictor:
    """Predictor and mitigation engine for backbreak in blast patterns."""

    def __init__(self):
        """Initialize backbreak predictor."""
        pass

    def predict_backbreak(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        burden_m: float,
        spacing_m: float,
        powder_factor_kg_m3: float,
        rock_factor_A: float = 8.0,
        inter_row_delay_ms: float = 42.0
    ) -> Dict[str, Any]:
        """
        Predict backbreak distance and highwall damage envelope.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            burden_m: Burden distance (m)
            spacing_m: Spacing distance (m)
            powder_factor_kg_m3: Powder factor (kg/m3)
            rock_factor_A: Rock blastability factor A
            inter_row_delay_ms: Delay between rows (ms)

        Returns:
            Dictionary with backbreak prediction and risk metrics.
        """
        # Base backbreak distance from modified Konya empirical model
        base_backbreak = burden_m * 0.28

        # Powder factor scaling (over-charged blasts exacerbate backbreak exponentially)
        pf_factor = (max(powder_factor_kg_m3, 0.2) / 0.65) ** 0.65

        # Rock structure factor
        rock_factor = max(rock_factor_A, 2.0) / 8.0

        # Burden-spacing stiffness ratio
        bs_ratio = burden_m / max(spacing_m, 0.1)
        bs_factor = 1.15 if bs_ratio > 1.1 else (0.95 if bs_ratio < 0.8 else 1.0)

        # Delay factor: fast row delays choke face and push energy backwards
        delay_factor = 1.25 if inter_row_delay_ms < 25.0 else (0.95 if inter_row_delay_ms >= 42.0 else 1.05)

        predicted_backbreak = base_backbreak * pf_factor * rock_factor * bs_factor * delay_factor
        backbreak_percentage = (predicted_backbreak / max(burden_m, 0.1)) * 100.0

        # Damage envelope depth (zone of micro-cracking behind back row)
        damage_envelope_depth_m = predicted_backbreak * 1.8

        if backbreak_percentage < 15.0:
            risk_level = "Low"
        elif backbreak_percentage < 25.0:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            "predicted_backbreak_m": round(predicted_backbreak, 2),
            "backbreak_percentage": round(backbreak_percentage, 1),
            "damage_envelope_depth_m": round(damage_envelope_depth_m, 2),
            "risk_level": risk_level,
            "acceptability": self._get_backbreak_acceptability(backbreak_percentage),
            "mitigation": self._get_backbreak_mitigation(risk_level),
        }

    def prevent_backbreak(
        self,
        predicted_backbreak_m: float,
        current_burden_m: float,
        current_spacing_m: float,
        current_pf_kg_m3: float,
        bench_height_m: float,
        current_stemming_m: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Calculate automated engineering mitigation parameters to prevent highwall backbreak.
        """
        # Recommended trim-row burden (reduce back row burden by 15-25%)
        trim_burden_m = round(current_burden_m * 0.80, 2)
        # Recommended trim-row spacing
        trim_spacing_m = round(current_spacing_m * 0.85, 2)
        # Recommended back-row charge reduction via air-deck or lower density explosive
        back_row_charge_reduction_pct = 30.0 if predicted_backbreak_m > 1.5 else 18.0
        # Recommended increased stemming on back row
        recommended_stemming_m = round(max(current_stemming_m * 1.20, bench_height_m * 0.35), 2)

        return {
            "recommended_trim_burden_m": trim_burden_m,
            "recommended_trim_spacing_m": trim_spacing_m,
            "back_row_charge_reduction_pct": back_row_charge_reduction_pct,
            "recommended_back_row_stemming_m": recommended_stemming_m,
            "recommended_back_row_delay_ms": 65,  # Extended delay for back row relief
            "presplit_recommended": predicted_backbreak_m > 2.0,
        }

    def _get_backbreak_acceptability(self, percentage: float) -> str:
        """Get backbreak acceptability string."""
        if percentage < 15.0:
            return "Acceptable — minimal backbreak expected, highwall stable"
        elif percentage < 25.0:
            return "Marginal — monitor crest fracturing and verify back row timing"
        else:
            return "Unacceptable — redesign back row with trim burden and air decking"

    def _get_backbreak_mitigation(self, risk_level: str) -> str:
        """Get backbreak mitigation recommendations."""
        mitigations = {
            "Low": "Maintain standard perimeter delay and verify collar positions",
            "Moderate": "Reduce back-row powder factor by 15-20% and extend row delay to 50ms+",
            "High": "Implement a trim row (0.8x burden), reduce back-row charge by 30%, or pre-split",
        }
        return mitigations.get(risk_level, "Review perimeter blast design")


class FreeFaceAnalyzer:
    """Analyzer for free face identification, orientation, and initiation relief vectors."""

    def __init__(self):
        """Initialize free face analyzer."""
        pass

    def identify_free_faces(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        face_data: Optional[List[Tuple[float, float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Identify free faces and classify holes based on free face exposure.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            face_data: Optional 3D face survey points

        Returns:
            Dictionary with free face analysis and classification of front row holes.
        """
        if not holes:
            return {"error": "No holes provided"}

        face_azimuth = bench_geometry.azimuth_deg
        face_dip = bench_geometry.face_angle_deg

        # Identify front row holes (closest to face Y coordinate)
        min_y = min(h.y for h in holes)
        front_row = [h for h in holes if h.y <= min_y + (h.burden_m * 0.75)]

        # Identify flank holes (extremes of X coordinates)
        min_x = min(h.x for h in holes)
        max_x = max(h.x for h in holes)
        left_flank = [h for h in holes if h.x <= min_x + (h.spacing_m * 0.4)]
        right_flank = [h for h in holes if h.x >= max_x - (h.spacing_m * 0.4)]

        # Relief direction (normal to face)
        relief_azimuth = (face_azimuth + 180.0) % 360.0

        return {
            "face_azimuth_deg": face_azimuth,
            "face_dip_deg": face_dip,
            "relief_direction_azimuth_deg": relief_azimuth,
            "front_row_holes": [h.hole_id for h in front_row],
            "num_front_row_holes": len(front_row),
            "left_flank_holes_count": len(left_flank),
            "right_flank_holes_count": len(right_flank),
            "total_holes": len(holes),
            "face_quality": self._assess_face_quality(bench_geometry.bench_height_m, face_dip),
        }

    def _assess_face_quality(self, height: float, dip: float) -> str:
        """Assess overall bench face quality."""
        if 65.0 <= dip <= 80.0:
            return "Optimal face angle (65°-80°) for forward displacement and safety"
        elif dip > 80.0:
            return "Steep face (>80°) — high risk of overhangs; monitor collar drill line"
        else:
            return "Flatter face (<65°) — large toe burden requires heavy subdrilling or angle drilling"


class CollisionDetector:
    """3D collision detector computing true minimum distance between 3D hole trajectories."""

    def __init__(self):
        """Initialize collision detector."""
        pass

    def detect_collisions(
        self,
        holes: List[BlastHole],
        min_separation_m: float = 1.2,
        warning_separation_m: float = 2.5
    ) -> Dict[str, Any]:
        """
        Detect hole-to-hole intersections and critical proximities in 3D space.

        Args:
            holes: List of blast holes
            min_separation_m: Minimum allowed physical separation between hole walls (m)
            warning_separation_m: Proximity warning threshold (m)

        Returns:
            Dictionary with collision and proximity analysis.
        """
        if not holes or len(holes) < 2:
            return {"num_collisions": 0, "collisions": [], "collision_free": True, "recommendation": "Pattern is collision-free"}

        collisions = []
        warnings = []

        for i in range(len(holes)):
            h1 = holes[i]
            # Collar and toe points for hole 1
            p1 = np.array([h1.x, h1.y, h1.collar_elevation])
            q1 = np.array([
                h1.x + h1.deviation_x,
                h1.y + h1.deviation_y,
                h1.toe_elevation
            ])
            r1 = (h1.hole_diameter_mm / 1000.0) / 2.0

            for j in range(i + 1, len(holes)):
                h2 = holes[j]
                p2 = np.array([h2.x, h2.y, h2.collar_elevation])
                q2 = np.array([
                    h2.x + h2.deviation_x,
                    h2.y + h2.deviation_y,
                    h2.toe_elevation
                ])
                r2 = (h2.hole_diameter_mm / 1000.0) / 2.0

                # Compute true 3D minimum distance between segment (p1->q1) and segment (p2->q2)
                min_dist, cp1, cp2, s, t = self.distance_between_segments_3d(p1, q1, p2, q2)
                effective_wall_sep = max(0.0, min_dist - (r1 + r2))

                if effective_wall_sep < min_separation_m:
                    collisions.append({
                        "hole1_id": h1.hole_id,
                        "hole2_id": h2.hole_id,
                        "min_separation_m": round(effective_wall_sep, 2),
                        "closest_depth_m": round(float(cp1[2] - h1.collar_elevation), 2),
                        "type": "Collision" if effective_wall_sep <= 0.05 else "Critical Proximity",
                    })
                elif effective_wall_sep < warning_separation_m:
                    warnings.append({
                        "hole1_id": h1.hole_id,
                        "hole2_id": h2.hole_id,
                        "min_separation_m": round(effective_wall_sep, 2),
                        "type": "Proximity Warning",
                    })

        return {
            "num_collisions": len(collisions),
            "num_warnings": len(warnings),
            "collisions": collisions,
            "warnings": warnings,
            "collision_free": len(collisions) == 0,
            "recommendation": self._get_collision_recommendation(len(collisions), len(warnings)),
        }

    @staticmethod
    def distance_between_segments_3d(
        p1: np.ndarray,
        q1: np.ndarray,
        p2: np.ndarray,
        q2: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
        """
        Calculate exact shortest distance between two 3D line segments S1(s) and S2(t).

        Returns:
            (distance, closest_pt_on_s1, closest_pt_on_s2, s_param, t_param)
        """
        u = q1 - p1
        v = q2 - p2
        w0 = p1 - p2

        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w0))
        e = float(np.dot(v, w0))

        denom = a * c - b * b

        if denom < 1e-8:
            s_c = 0.0
            t_c = d / b if b > 1e-8 else 0.0
        else:
            s_c = (b * e - c * d) / denom
            t_c = (a * e - b * d) / denom

        # Clamp to line segment boundaries [0, 1]
        s_c = float(np.clip(s_c, 0.0, 1.0))
        t_c = float(np.clip(t_c, 0.0, 1.0))

        # Re-resolve closest point after clamping
        pt1 = p1 + s_c * u
        pt2 = p2 + t_c * v
        dist = float(np.linalg.norm(pt1 - pt2))

        return dist, pt1, pt2, s_c, t_c

    def _get_collision_recommendation(self, num_collisions: int, num_warnings: int) -> str:
        """Get collision recommendation."""
        if num_collisions == 0 and num_warnings == 0:
            return "No collisions or critical proximities detected — design is safe to drill"
        elif num_collisions == 0:
            return f"{num_warnings} proximity warning(s) detected — review hole spacing"
        else:
            return f"🚨 {num_collisions} collision(s) detected — redesign hole positions to prevent drill bit collision"


class TopographyModeler:
    """Modeler for realistic bench face topography, crest curvature, and burden profiles."""

    def __init__(self):
        """Initialize topography modeler."""
        pass

    def create_topographic_surface(
        self,
        bench_geometry: BenchGeometry,
        roughness_factor: float = 0.5,
        num_points: int = 64,
        curvature_amplitude_m: float = 1.5
    ) -> List[Tuple[float, float, float]]:
        """
        Create 3D topographic surface representing the bench face with curvature and roughness.
        """
        points = []
        n_side = max(4, int(np.sqrt(num_points)))
        x_range = np.linspace(0, bench_geometry.bench_length_m, n_side)
        z_range = np.linspace(bench_geometry.toe_elevation, bench_geometry.crest_elevation, n_side)

        face_angle_rad = np.radians(max(bench_geometry.face_angle_deg, 30.0))

        for x in x_range:
            # Curvature along bench strike (e.g. slight pit bow)
            norm_x = x / max(bench_geometry.bench_length_m, 1.0)
            crest_curvature = curvature_amplitude_m * np.sin(np.pi * norm_x)

            for z in z_range:
                # Base Y offset based on elevation drop from crest
                depth_from_crest = bench_geometry.crest_elevation - z
                base_y = depth_from_crest / np.tan(face_angle_rad)

                # Synthetic rock roughness
                roughness = float(np.sin(x * 0.8) * np.cos(z * 0.5)) * roughness_factor * 0.6
                y = base_y + crest_curvature + roughness
                points.append((float(x), float(y), float(z)))

        return points

    def calculate_face_burden_profile(
        self,
        hole: BlastHole,
        bench_geometry: BenchGeometry,
        roughness_factor: float = 0.5,
        num_elevation_samples: int = 15,
    ) -> Dict[str, Any]:
        """
        Calculate continuous 2D/3D burden profile along the length of a blast hole from collar to toe.
        """
        face_angle_rad = np.radians(max(bench_geometry.face_angle_deg, 30.0))
        z_samples = np.linspace(hole.collar_elevation, hole.toe_elevation, num_elevation_samples)

        depths = []
        burdens = []
        for z in z_samples:
            depth = hole.collar_elevation - z
            # Interpolated hole Y position at this depth
            frac = depth / max(hole.hole_length, 0.1)
            hole_y = hole.y + frac * hole.deviation_y

            # Bench face Y position at elevation z
            face_y = depth / np.tan(face_angle_rad) + (roughness_factor * float(np.sin(depth * 0.4)))
            burden_at_z = max(0.1, hole_y + face_y)

            depths.append(round(float(depth), 2))
            burdens.append(round(float(burden_at_z), 2))

        min_b = min(burdens)
        max_b = max(burdens)

        return {
            "hole_id": hole.hole_id,
            "depths_m": depths,
            "burdens_m": burdens,
            "min_burden_m": round(min_b, 2),
            "max_burden_m": round(max_b, 2),
            "has_toe_bulge": burdens[-1] > (burdens[0] * 1.3),
        }

    def analyze_topography(
        self,
        topography_points: List[Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze topographic surface variance, roughness, and quality.
        """
        if not topography_points:
            return {"error": "No topography points provided"}

        y_coords = [p[1] for p in topography_points]
        y_range = max(y_coords) - min(y_coords)
        y_std = float(np.std(y_coords))

        roughness_index = y_std / max(y_range, 0.01)

        return {
            "y_range_m": round(y_range, 2),
            "y_std_m": round(y_std, 2),
            "roughness_index": round(roughness_index, 3),
            "surface_quality": self._assess_surface_quality(roughness_index),
        }

    def _assess_surface_quality(self, roughness_index: float) -> str:
        """Assess surface quality based on roughness index."""
        if roughness_index < 0.12:
            return "Very smooth face — predictable burden distribution"
        elif roughness_index < 0.22:
            return "Uniform face — typical open pit conditions"
        elif roughness_index < 0.32:
            return "Moderately rough face — localized burden variance possible"
        else:
            return "Irregular / heavily jointed face — continuous 3D profiling recommended"


class ConstraintBasedHolePlacer:
    """Automated hole placement engine constrained by bench boundaries and blasting guidelines."""

    def __init__(self):
        """Initialize constraint-based automated hole placer."""
        pass

    def place_holes_with_constraints(
        self,
        bench_geometry: BenchGeometry,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[BlastHole], Dict[str, Any]]:
        """
        Automatically place blast holes within bench boundaries honoring all constraints.

        Constraints supported:
            - min_burden_m, max_burden_m, nominal_burden_m
            - min_spacing_m, max_spacing_m, nominal_spacing_m
            - crest_standoff_m (min distance from crest to front row collars)
            - toe_standoff_m (clearance from bench toe)
            - rear_standoff_m (clearance from rear boundary / back highwall)
            - lateral_standoff_m (clearance from left/right side boundaries)
            - hole_diameter_mm
            - target_powder_factor_kg_m3
            - pattern_type: 'rectangular', 'staggered', 'echelon', 'v_pattern'
            - subdrilling_m
        """
        c = constraints or {}

        nom_burden = float(c.get("nominal_burden_m", 6.0))
        min_burden = float(c.get("min_burden_m", 4.0))
        max_burden = float(c.get("max_burden_m", 9.0))

        nom_spacing = float(c.get("nominal_spacing_m", 7.0))
        min_spacing = float(c.get("min_spacing_m", 5.0))
        max_spacing = float(c.get("max_spacing_m", 11.0))

        crest_standoff = float(c.get("crest_standoff_m", 3.5))
        rear_standoff = float(c.get("rear_standoff_m", 3.0))
        lateral_standoff = float(c.get("lateral_standoff_m", 3.0))
        hole_diameter_mm = float(c.get("hole_diameter_mm", 250.0))
        subdrilling_m = float(c.get("subdrilling_m", 1.5))
        powder_factor_kg_m3 = float(c.get("target_powder_factor_kg_m3", 0.65))
        pattern_type_str = str(c.get("pattern_type", "staggered")).lower()

        # Calculate active drilling rectangle
        usable_x_min = lateral_standoff
        usable_x_max = bench_geometry.bench_length_m - lateral_standoff
        usable_y_min = crest_standoff
        usable_y_max = bench_geometry.bench_width_m - rear_standoff

        usable_x_span = max(0.1, usable_x_max - usable_x_min)
        usable_y_span = max(0.1, usable_y_max - usable_y_min)

        # Solve for optimal row and column count
        # Rows span Y direction (burden)
        ideal_rows = max(2, int(round(usable_y_span / nom_burden)) + 1)
        actual_burden = usable_y_span / max(1, ideal_rows - 1)
        actual_burden = float(np.clip(actual_burden, min_burden, max_burden))
        num_rows = int(np.clip(ideal_rows, 2, 15))

        # Columns span X direction (spacing)
        ideal_cols = max(2, int(round(usable_x_span / nom_spacing)) + 1)
        actual_spacing = usable_x_span / max(1, ideal_cols - 1)
        actual_spacing = float(np.clip(actual_spacing, min_spacing, max_spacing))
        num_cols = int(np.clip(ideal_cols, 2, 20))

        stemming_m = round(actual_burden * 0.70, 2)
        hole_length = bench_geometry.bench_height_m + subdrilling_m

        # Calculate charge per hole
        area_per_hole = actual_burden * actual_spacing
        volume_per_hole = area_per_hole * bench_geometry.bench_height_m
        charge_mass_kg = round(volume_per_hole * powder_factor_kg_m3, 1)

        holes: List[BlastHole] = []
        hole_idx = 1

        for row in range(num_rows):
            y_pos = usable_y_min + (row * actual_burden)

            # Pattern offset calculations
            x_offset = 0.0
            if "stagger" in pattern_type_str and (row % 2 == 1):
                x_offset = actual_spacing * 0.5
            elif "echelon" in pattern_type_str:
                x_offset = row * float(c.get("echelon_offset_m", 2.0))

            for col in range(num_cols):
                base_x = usable_x_min + (col * actual_spacing)

                if "v_pattern" in pattern_type_str:
                    # Chevron shaped apex
                    center_col = (num_cols - 1) / 2.0
                    v_y_shift = abs(col - center_col) * 0.5 * actual_burden
                    x_pos = base_x
                    curr_y = y_pos + v_y_shift
                else:
                    x_pos = base_x + x_offset
                    curr_y = y_pos

                # Check that hole stays inside lateral boundaries
                if x_pos > (bench_geometry.bench_length_m - lateral_standoff + 0.1):
                    continue
                if curr_y > (bench_geometry.bench_width_m - rear_standoff + 0.1):
                    continue

                collar_z = bench_geometry.crest_elevation
                toe_z = bench_geometry.toe_elevation - subdrilling_m

                hole = BlastHole(
                    hole_id=f"AUTO_H{hole_idx:03d}",
                    x=round(float(x_pos), 2),
                    y=round(float(curr_y), 2),
                    z=round(float(collar_z), 2),
                    collar_elevation=round(float(collar_z), 2),
                    toe_elevation=round(float(toe_z), 2),
                    hole_length=round(float(hole_length), 2),
                    hole_diameter_mm=hole_diameter_mm,
                    burden_m=round(actual_burden, 2),
                    spacing_m=round(actual_spacing, 2),
                    stemming_m=stemming_m,
                    powder_factor_kg_m3=powder_factor_kg_m3,
                    charge_mass_kg=charge_mass_kg,
                    deck_info=None,
                    deviation_x=0.0,
                    deviation_y=0.0,
                    deviation_z=0.0,
                )
                holes.append(hole)
                hole_idx += 1

        # Verify automated placement report
        total_drilled_m = round(len(holes) * hole_length, 1)
        total_charge_kg = round(len(holes) * charge_mass_kg, 1)
        total_volume_m3 = round(len(holes) * volume_per_hole, 1)
        tonnage = round(total_volume_m3 * 2.65, 1)  # 2.65 t/m3 standard rock density

        report = {
            "total_holes_placed": len(holes),
            "rows_count": num_rows,
            "cols_count": num_cols,
            "calculated_burden_m": round(actual_burden, 2),
            "calculated_spacing_m": round(actual_spacing, 2),
            "spacing_to_burden_ratio": round(actual_spacing / max(actual_burden, 0.1), 2),
            "crest_standoff_m": crest_standoff,
            "rear_standoff_m": rear_standoff,
            "lateral_standoff_m": lateral_standoff,
            "total_drilled_meters": total_drilled_m,
            "total_explosive_mass_kg": total_charge_kg,
            "expected_rock_tonnage": tonnage,
            "boundary_compliance": True,
        }

        return holes, report
