"""
Diamond Population Economics & Value Distribution Calculator.

Computes observed carat weight distribution, damaged vs undamaged valuations,
and scenario cost-benefit impacts with explicit valuation schedules.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from src.diamond_preservation.models import (
    DiamondObservation,
    DiamondValuationSchedule,
)


def calculate_population_valuation(
    observations: List[DiamondObservation],
    valuation_schedule: DiamondValuationSchedule,
) -> Dict[str, Any]:
    """
    Calculates total carats, total estimated value, and breakage value loss by size class.
    """
    if not observations:
        return {
            "total_stones": 0,
            "total_carats": 0.0,
            "total_estimated_value_usd": 0.0,
            "damaged_stones_count": 0,
            "damaged_carats": 0.0,
            "value_loss_from_damage_usd": 0.0,
            "class_breakdown": [],
            "schedule_name": valuation_schedule.schedule_name,
            "currency": valuation_schedule.currency,
        }

    v_map = valuation_schedule.valuation_map

    total_stones = 0
    total_carats = 0.0
    total_val = 0.0
    damaged_stones = 0
    damaged_carats = 0.0
    estimated_loss = 0.0

    class_stats: Dict[str, Dict[str, Any]] = {}

    for obs in observations:
        total_stones += obs.stone_count
        total_carats += obs.total_carats

        # Base price per carat
        base_price = v_map.get(obs.stone_size_class, v_map.get(f"{obs.stone_size_class} {obs.stone_quality_category}", 100.0))
        obs_val = obs.total_carats * base_price
        total_val += obs_val

        # Loss estimation if damaged
        loss_pct = 0.0
        if obs.damage_type == "chipping":
            loss_pct = 0.15
        elif obs.damage_type == "cracking":
            loss_pct = 0.35
        elif obs.damage_type in ["breakage_cleavage", "severe_fragmentation"]:
            loss_pct = 0.65

        if obs.damage_type != "none":
            damaged_stones += obs.stone_count
            damaged_carats += obs.total_carats
            loss_val = obs_val * loss_pct
            estimated_loss += loss_val

        s_cls = obs.stone_size_class
        if s_cls not in class_stats:
            class_stats[s_cls] = {
                "size_class": s_cls,
                "stone_count": 0,
                "total_carats": 0.0,
                "estimated_value_usd": 0.0,
            }
        class_stats[s_cls]["stone_count"] += obs.stone_count
        class_stats[s_cls]["total_carats"] += obs.total_carats
        class_stats[s_cls]["estimated_value_usd"] += obs_val

    return {
        "total_stones": total_stones,
        "total_carats": round(total_carats, 2),
        "total_estimated_value_usd": round(total_val, 2),
        "damaged_stones_count": damaged_stones,
        "damaged_carats": round(damaged_carats, 2),
        "value_loss_from_damage_usd": round(estimated_loss, 2),
        "class_breakdown": list(class_stats.values()),
        "schedule_name": valuation_schedule.schedule_name,
        "currency": valuation_schedule.currency,
    }
