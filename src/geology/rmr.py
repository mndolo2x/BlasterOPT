"""
Rock Mass Rating (RMR89) classification system (Bieniawski, 1989).
"""

from typing import Dict, Any, Optional


class RMRCalculator:
    """
    Rock Mass Rating (Bieniawski 1989, "Engineering Rock Mass Classifications").

    RMR = UCS_rating + RQD_rating + Spacing_rating
          + Condition_rating + Groundwater_rating + Orientation_rating

    Final RMR ranges from 0 to 100.
    """

    def calculate_ucs_rating(self, ucs_mpa: float) -> int:
        if ucs_mpa < 0:
            raise ValueError(f"ucs_mpa ({ucs_mpa}) cannot be negative")

        if ucs_mpa > 250.0:
            return 15
        elif ucs_mpa >= 100.0:
            return 12
        elif ucs_mpa >= 50.0:
            return 7
        elif ucs_mpa >= 25.0:
            return 4
        elif ucs_mpa >= 5.0:
            return 2
        elif ucs_mpa >= 1.0:
            return 1
        else:
            return 0

    def calculate_rqd_rating(self, rqd_pct: float) -> int:
        if not (0.0 <= rqd_pct <= 100.0):
            raise ValueError(f"rqd_pct ({rqd_pct}) must be between 0 and 100")

        if rqd_pct >= 90.0:
            return 20
        elif rqd_pct >= 75.0:
            return 17
        elif rqd_pct >= 50.0:
            return 13
        elif rqd_pct >= 25.0:
            return 8
        else:
            return 3

    def calculate_spacing_rating(self, joint_spacing_m: float) -> int:
        if joint_spacing_m < 0:
            raise ValueError(f"joint_spacing_m ({joint_spacing_m}) cannot be negative")

        if joint_spacing_m > 2.0:
            return 20
        elif joint_spacing_m >= 0.6:
            return 15
        elif joint_spacing_m >= 0.2:
            return 10
        elif joint_spacing_m >= 0.06:
            return 8
        else:
            return 5

    def calculate_condition_rating(
        self,
        persistence_m: float,
        aperture_mm: float,
        roughness: str,
        infilling: str,
        weathering: str,
    ) -> int:
        if persistence_m < 0 or aperture_mm < 0:
            raise ValueError("persistence_m and aperture_mm cannot be negative")

        rough = roughness.lower()
        inf = infilling.lower()
        weath = weathering.lower()

        if "gouge >5" in inf or infilling.lower() == "thick_gouge":
            return 0
        elif "slickensided" in rough or "gouge <5" in inf or infilling.lower() == "thin_gouge":
            return 10
        elif "highly_weathered" in weath or aperture_mm >= 1.0:
            return 20
        elif "slightly" in rough or "slightly" in weath or aperture_mm < 1.0:
            return 25
        elif "very_rough" in rough and "unweathered" in weath and aperture_mm == 0:
            return 30
        else:
            return 20

    def calculate_groundwater_rating(
        self, inflow_l_min: float, joint_water_pressure_mpa: float
    ) -> int:
        if inflow_l_min == 0.0 and joint_water_pressure_mpa == 0.0:
            return 15
        elif inflow_l_min < 10.0 and joint_water_pressure_mpa < 0.1:
            return 10
        elif inflow_l_min < 25.0 and joint_water_pressure_mpa < 0.2:
            return 7
        elif inflow_l_min < 125.0 and joint_water_pressure_mpa < 0.5:
            return 4
        else:
            return 0

    def calculate_orientation_rating(
        self, joint_orientation_deg: float, tunnel_azimuth_deg: float
    ) -> int:
        angle_diff = abs((joint_orientation_deg - tunnel_azimuth_deg + 180.0) % 360.0 - 180.0)

        if angle_diff < 20.0 or angle_diff > 160.0:
            return 0
        elif angle_diff < 45.0 or angle_diff > 135.0:
            return -2
        elif angle_diff < 70.0 or angle_diff > 110.0:
            return -7
        else:
            return -12

    def calculate_total(
        self,
        ucs_mpa: float,
        rqd_pct: float,
        joint_spacing_m: float,
        persistence_m: float = 1.0,
        aperture_mm: float = 0.5,
        roughness: str = "slightly_rough",
        infilling: str = "none",
        weathering: str = "slightly_weathered",
        inflow_l_min: float = 0.0,
        joint_water_pressure_mpa: float = 0.0,
        joint_orientation_deg: float = 0.0,
        tunnel_azimuth_deg: float = 0.0,
    ) -> Dict[str, Any]:
        r_ucs = self.calculate_ucs_rating(ucs_mpa)
        r_rqd = self.calculate_rqd_rating(rqd_pct)
        r_spacing = self.calculate_spacing_rating(joint_spacing_m)
        r_cond = self.calculate_condition_rating(persistence_m, aperture_mm, roughness, infilling, weathering)
        r_gw = self.calculate_groundwater_rating(inflow_l_min, joint_water_pressure_mpa)
        r_orient = self.calculate_orientation_rating(joint_orientation_deg, tunnel_azimuth_deg)

        total_rmr = max(0, min(100, r_ucs + r_rqd + r_spacing + r_cond + r_gw + r_orient))

        if total_rmr > 80:
            cls = "very_good"
            support = "Full face excavation, 3m advance. Generally no support required except occasional spot bolts."
            max_span = 15.0
            stand_up = "20 years for 15 m span"
        elif total_rmr > 60:
            cls = "good"
            support = "Full face excavation, 1.0-1.5m advance. Complete support 20m from face. Spot bolting."
            max_span = 10.0
            stand_up = "1 year for 10 m span"
        elif total_rmr > 40:
            cls = "fair"
            support = "Top heading and bench, 1.5-3.0m advance. Systematic bolting 1.5-2.0m spacing with wire mesh."
            max_span = 5.0
            stand_up = "1 week for 5 m span"
        elif total_rmr > 20:
            cls = "poor"
            support = "Top heading and bench, 1.0-1.5m advance. Systematic bolting 1.0-1.5m spacing with 50-100mm shotcrete."
            max_span = 2.5
            stand_up = "10 hours for 2.5 m span"
        else:
            cls = "very_poor"
            support = "Multiple drifts, 0.5-1.0m advance. Medium to heavy steel ribs with continuous lagging and 100-150mm shotcrete."
            max_span = 1.0
            stand_up = "30 minutes for 1 m span"

        return {
            "rmr": total_rmr,
            "class": cls,
            "components": {
                "ucs": r_ucs,
                "rqd": r_rqd,
                "spacing": r_spacing,
                "condition": r_cond,
                "groundwater": r_gw,
                "orientation": r_orient,
            },
            "excavation_support": support,
            "max_span_m": max_span,
            "stand_up_time": stand_up,
        }


def calculate_rmr89(
    ucs_mpa: float,
    rqd_pct: float,
    joint_spacing_m: float,
    joint_condition: str,
    groundwater_condition: str,
    joint_orientation_rating: float = 0.0,
) -> Dict[str, Any]:
    calc = RMRCalculator()
    res = calc.calculate_total(
        ucs_mpa=ucs_mpa,
        rqd_pct=rqd_pct,
        joint_spacing_m=joint_spacing_m,
        roughness=joint_condition,
    )
    return {
        "rmr_score": res["rmr"],
        "rock_class": res["class"],
        "breakdown": res["components"],
    }
