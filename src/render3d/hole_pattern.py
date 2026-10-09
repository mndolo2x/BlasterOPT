"""
3D drill hole pattern generation for four pattern types.
"""
import numpy as np
from typing import List, Literal, Tuple, Dict, Any
from pydantic import BaseModel, Field


class Hole(BaseModel):
    """Single blasthole in 3D space."""
    hole_id: str
    row: int
    column: int
    x_m: float
    y_m: float
    z_collar_m: float
    z_toe_m: float
    depth_m: float
    angle_deg: float
    charge_kg: float
    stemming_m: float
    delay_ms: float


class HolePattern(BaseModel):
    """Complete drill pattern."""
    pattern_type: Literal["square", "staggered", "echelon", "v_pattern"]
    burden_m: float
    spacing_m: float
    num_rows: int
    holes_per_row: int
    stemming_m: float
    hole_depth_m: float
    hole_angle_deg: float
    powder_factor_kg_m3: float
    bench_height_m: float
    subdrilling_m: float
    holes: List[Hole] = Field(default_factory=list)


def _compute_charge_per_hole(pattern: HolePattern) -> float:
    """
    Compute explosive mass per hole.

    Formula:
        rock_volume_per_hole = burden × spacing × bench_height
        charge_per_hole = powder_factor × rock_volume_per_hole
    """
    volume = pattern.burden_m * pattern.spacing_m * pattern.bench_height_m
    return float(pattern.powder_factor_kg_m3 * volume)


def _compute_position(
    pattern_type: str, row: int, col: int, pattern: HolePattern
) -> Tuple[float, float]:
    """
    Return (x, y) coordinates for a hole based on the pattern type.
    """
    x_base = col * pattern.spacing_m
    y_base = row * pattern.burden_m

    if pattern_type == "square":
        return x_base, y_base

    if pattern_type == "staggered":
        x = x_base + (row % 2) * pattern.spacing_m / 2.0
        return x, y_base

    if pattern_type == "echelon":
        x = x_base + row * pattern.spacing_m * 0.25
        return x, y_base

    if pattern_type == "v_pattern":
        mid = pattern.num_rows / 2.0
        offset = abs(row - mid) * pattern.spacing_m * 0.3
        x = x_base + offset
        return x, y_base

    raise ValueError(f"Unknown pattern type: {pattern_type}")


def generate_holes(
    pattern: HolePattern,
    inter_hole_delay_ms: float = 20.0,
    inter_row_delay_ms: float = 60.0,
) -> HolePattern:
    """
    Populate pattern.holes with Hole objects.

    Hole IDs follow the format H-R{row}-C{col}.
    Delay assigns inter-hole delay within a row and inter-row delay between rows.
    """
    charge = _compute_charge_per_hole(pattern)
    holes: List[Hole] = []

    for row in range(pattern.num_rows):
        for col in range(pattern.holes_per_row):
            x, y = _compute_position(pattern.pattern_type, row, col, pattern)
            z_collar = 0.0  # local bench coordinate
            z_toe = -pattern.hole_depth_m
            delay = row * inter_row_delay_ms + col * inter_hole_delay_ms

            holes.append(Hole(
                hole_id=f"H-R{row}-C{col}",
                row=row,
                column=col,
                x_m=float(x),
                y_m=float(y),
                z_collar_m=float(z_collar),
                z_toe_m=float(z_toe),
                depth_m=pattern.hole_depth_m,
                angle_deg=pattern.hole_angle_deg,
                charge_kg=charge,
                stemming_m=pattern.stemming_m,
                delay_ms=float(delay),
            ))

    pattern.holes = holes
    return pattern


def pattern_summary(pattern: HolePattern) -> Dict[str, Any]:
    """Return a summary of the pattern for display."""
    if not pattern.holes:
        raise ValueError("Pattern has no holes. Call generate_holes() first.")
    total_charge = sum(h.charge_kg for h in pattern.holes)
    return {
        "num_holes": len(pattern.holes),
        "num_rows": pattern.num_rows,
        "holes_per_row": pattern.holes_per_row,
        "total_charge_kg": total_charge,
        "max_delay_ms": max(h.delay_ms for h in pattern.holes),
        "min_delay_ms": min(h.delay_ms for h in pattern.holes),
        "pattern_type": pattern.pattern_type,
    }
