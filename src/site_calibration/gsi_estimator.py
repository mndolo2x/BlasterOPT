"""
Geological Strength Index (GSI) Estimator Submodule.
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Configurable reference table for rock type default GSI values
ROCK_TYPE_GSI_TABLE: Dict[str, float] = {
    "kimberlite_competent": 65.0,
    "kimberlite_weathered": 45.0,
    "kimberlite": 60.0,
    "waste_granite_hard": 75.0,
    "granite": 70.0,
    "sandstone_medium": 55.0,
    "sandstone": 50.0,
    "shale_soft": 35.0,
    "basalt": 68.0,
    "dolerite": 72.0,
}


class GSIEstimator:
    """
    Estimates Geological Strength Index (GSI) from multiple geological data sources:
    1. Direct GSI measurement input
    2. Bieniawski Rock Mass Rating (RMR) conversion: GSI ≈ RMR - 5 (for RMR > 23)
    3. Rock type lookup table
    4. Geological block model spatial interpolation
    """

    def __init__(self, custom_rock_table: Optional[Dict[str, float]] = None):
        self.rock_table = custom_rock_table or ROCK_TYPE_GSI_TABLE

    def estimate_gsi(
        self,
        direct_gsi: Optional[float] = None,
        rmr: Optional[float] = None,
        rock_type: Optional[str] = None,
        block_model_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], float]:
        """
        Estimates GSI and returns (gsi_value, confidence_score_0_to_1).

        Returns:
        --------
        Tuple[Optional[float], float]
            (GSI value in range [0, 100], confidence score between 0.0 and 1.0).
            If no GSI data can be derived, returns (None, 0.0).
        """
        # 1. Direct GSI input (Highest confidence = 0.95)
        if direct_gsi is not None and 0 <= direct_gsi <= 100:
            return float(direct_gsi), 0.95

        # 2. RMR Conversion: GSI ≈ RMR - 5 for RMR > 23 (Confidence = 0.85)
        if rmr is not None:
            rmr_float = float(rmr)
            if rmr_float > 23:
                calc_gsi = max(0.0, min(100.0, rmr_float - 5.0))
                return calc_gsi, 0.85
            elif 0 <= rmr_float <= 23:
                # Hoek & Brown lower bound rule
                return 18.0, 0.70

        # 3. Geological block model spatial interpolation along path (Confidence = 0.80)
        if block_model_data and "path_gsi_average" in block_model_data:
            path_gsi = float(block_model_data["path_gsi_average"])
            if 0 <= path_gsi <= 100:
                return path_gsi, 0.80

        # 4. Rock type lookup table (Confidence = 0.60)
        if rock_type:
            rt_key = rock_type.lower().strip().replace(" ", "_")
            if rt_key in self.rock_table:
                return self.rock_table[rt_key], 0.60
            for k, v in self.rock_table.items():
                if k in rt_key or rt_key in k:
                    return v, 0.55

        # Fallback to None if no geological data is available
        return None, 0.0
