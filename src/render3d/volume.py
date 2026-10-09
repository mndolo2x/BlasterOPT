"""
Volume and tonnage calculation from the 3D pattern.
"""
from src.render3d.hole_pattern import HolePattern
from src.render3d.constants import ROCK_DENSITY_T_M3


def compute_blast_volume(pattern: HolePattern) -> float:
    """
    Compute in-situ rock volume covered by the pattern.

    Formula:
        volume = burden × spacing × bench_height × num_holes
    """
    if not pattern.holes:
        raise ValueError("Pattern has no holes.")
    volume_per_hole = pattern.burden_m * pattern.spacing_m * pattern.bench_height_m
    return float(volume_per_hole * len(pattern.holes))


def compute_blast_tonnage(pattern: HolePattern, rock_density: float = ROCK_DENSITY_T_M3) -> float:
    """
    Compute tonnage from volume and rock density.

    Formula:
        tonnage = volume_m3 × rock_density_t_m3
    """
    return compute_blast_volume(pattern) * rock_density


def compute_powder_factor_from_charge(
    pattern: HolePattern, total_charge_kg: float
) -> float:
    """
    Back-calculate powder factor from a known total charge.

    Formula:
        pf = total_charge_kg / volume_m3
    """
    volume = compute_blast_volume(pattern)
    if volume <= 0:
        raise ValueError("Volume must be positive.")
    return float(total_charge_kg / volume)
