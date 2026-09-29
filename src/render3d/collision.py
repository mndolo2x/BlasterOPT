"""
3D hole-to-hole collision detection for BlasterOPT 3D engine.
"""

import math
import numpy as np
from typing import List, Dict, Any
from src.render3d.hole_pattern import Hole


def check_hole_collisions(
    holes: List[Hole],
    min_separation_m: float = 0.5,
    max_deviation_deg: float = 3.0,
) -> Dict[str, Any]:
    """
    Check for hole-to-hole collisions considering drilling deviation.

    Two holes collide if:
        distance_at_toe < min_separation_m
        where distance_at_toe accounts for angular deviation over hole depth.

    Deviation model:
        horizontal_deviation = depth * tan(deviation_radians)

    Returns:
        {
            "has_collisions": bool,
            "collisions": [
                {
                    "hole_a": str,
                    "hole_b": str,
                    "surface_distance_m": float,
                    "toe_distance_m": float,
                    "risk": str,  # 'critical', 'warning', 'safe'
                }
            ],
            "recommendations": list[str],
        }
    """
    if min_separation_m <= 0:
        raise ValueError(f"min_separation_m ({min_separation_m}) must be positive")
    if max_deviation_deg < 0:
        raise ValueError(f"max_deviation_deg ({max_deviation_deg}) cannot be negative")

    collisions: List[Dict[str, Any]] = []
    has_collisions = False
    recommendations: List[str] = []

    dev_rad = math.radians(max_deviation_deg)

    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            h_a = holes[i]
            h_b = holes[j]

            # Surface collar distance
            # Formula: surface_distance = sqrt((x_a - x_b)^2 + (y_a - y_b)^2 + (z_a - z_b)^2)
            surf_dist = math.sqrt((h_a.x - h_b.x) ** 2 + (h_a.y - h_b.y) ** 2 + (h_a.z - h_b.z) ** 2)

            # Toe coordinates
            toe_a = h_a.get_toe_coordinate()
            toe_b = h_b.get_toe_coordinate()

            # Formula: horizontal_deviation = depth * tan(deviation_radians)
            dev_a = h_a.depth_m * math.tan(dev_rad)
            dev_b = h_b.depth_m * math.tan(dev_rad)

            # Nominal toe distance
            nom_toe_dist = math.sqrt(
                (toe_a[0] - toe_b[0]) ** 2 + (toe_a[1] - toe_b[1]) ** 2 + (toe_a[2] - toe_b[2]) ** 2
            )

            # Worst-case toe distance accounting for angular deviation
            worst_toe_dist = max(0.0, nom_toe_dist - (dev_a + dev_b))

            if worst_toe_dist < min_separation_m:
                risk = "critical"
                has_collisions = True
            elif worst_toe_dist < (min_separation_m * 2.0):
                risk = "warning"
            else:
                risk = "safe"

            if risk in ("critical", "warning"):
                collisions.append(
                    {
                        "hole_a": h_a.hole_id,
                        "hole_b": h_b.hole_id,
                        "surface_distance_m": round(surf_dist, 2),
                        "toe_distance_m": round(worst_toe_dist, 2),
                        "risk": risk,
                    }
                )

    if has_collisions:
        recommendations.append(
            f"Critical collision risk detected (<{min_separation_m}m separation). Adjust collar positions or hole dip angles."
        )
        recommendations.append("Use MWD gyro logging or drill rig inclination sensors to track actual hole trajectory.")
    elif len(collisions) > 0:
        recommendations.append("Warning collision risk detected. Verify collar spacing and drill rig setup.")

    return {
        "has_collisions": has_collisions,
        "collisions": collisions,
        "recommendations": recommendations,
    }
