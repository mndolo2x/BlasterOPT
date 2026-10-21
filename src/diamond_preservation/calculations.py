"""
Engineering Calculations Service for Diamond Preservation.

Provides verified formulas for:
- Specific Charge (kg explosive / tonne rock)
- Drilling Depth Deviation (m and %)
- Charge Distribution Density (kg explosive / m hole depth)
- Specific Energy (MJ / m^3)
- Scaled Vibration Distance (m / sqrt(kg))
"""

import numpy as np
from typing import Dict, Any, Optional


def calculate_specific_charge(
    total_explosive_kg: float,
    blasted_tonnage: float,
) -> Dict[str, Any]:
    """
    Calculates specific charge (kg/t).

    Formula:
        specific_charge = total_explosive_kg / blasted_tonnage
    """
    if blasted_tonnage <= 0 or total_explosive_kg <= 0:
        return {
            "value": None,
            "unit": "kg/t",
            "formula": "total_explosive_kg / blasted_tonnage",
            "error": "Tonnage and explosive mass must be greater than zero.",
        }

    sc = total_explosive_kg / blasted_tonnage
    return {
        "value": round(sc, 4),
        "unit": "kg/t",
        "formula": "total_explosive_kg / blasted_tonnage",
        "inputs": {"total_explosive_kg": total_explosive_kg, "blasted_tonnage": blasted_tonnage},
    }


def calculate_drilling_deviation(
    planned_depth_m: float,
    actual_depth_m: float,
) -> Dict[str, Any]:
    """
    Calculates drilling depth deviation (m and %).
    """
    if planned_depth_m <= 0:
        return {
            "depth_delta_m": None,
            "percentage_deviation": None,
            "error": "Planned depth must be > 0",
        }

    delta_m = actual_depth_m - planned_depth_m
    pct = (delta_m / planned_depth_m) * 100.0

    return {
        "planned_depth_m": planned_depth_m,
        "actual_depth_m": actual_depth_m,
        "depth_delta_m": round(delta_m, 2),
        "percentage_deviation": round(pct, 2),
        "unit": "m",
    }


def calculate_charge_density(
    total_explosive_kg: float,
    total_hole_depth_m: float,
) -> Dict[str, Any]:
    """
    Calculates linear charge loading density (kg/m).
    """
    if total_hole_depth_m <= 0:
        return {"value": None, "unit": "kg/m", "error": "Hole depth must be > 0"}

    density = total_explosive_kg / total_hole_depth_m
    return {
        "value": round(density, 2),
        "unit": "kg/m",
        "formula": "total_explosive_kg / total_hole_depth_m",
    }


def calculate_scaled_distance(
    distance_m: float,
    max_charge_per_delay_kg: float,
) -> Dict[str, Any]:
    """
    Calculates USBM scaled distance SD = distance / sqrt(charge_per_delay).
    """
    if max_charge_per_delay_kg <= 0 or distance_m <= 0:
        return {"value": None, "unit": "m/kg^0.5", "error": "Distance and charge must be > 0"}

    sd = distance_m / np.sqrt(max_charge_per_delay_kg)
    return {
        "value": round(sd, 2),
        "unit": "m/kg^0.5",
        "formula": "distance_m / sqrt(max_charge_per_delay_kg)",
    }
