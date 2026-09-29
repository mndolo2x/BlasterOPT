"""
Rock mass Q-system classification system (Barton et al., 1974).
"""

from typing import Dict, Any, List


class QSystemCalculator:
    """
    Q-system (Barton et al. 1974, "Engineering Classification of Rock Masses").

    Q = (RQD / Jn) × (Jr / Ja) × (Jw / SRF)

    Values range from 0.001 (exceptionally poor) to 1000 (exceptionally good).
    """

    def calculate_rqd(self, core_recoveries: List[float], total_length_m: float = 1.0) -> float:
        if total_length_m <= 0:
            raise ValueError(f"total_length_m ({total_length_m}) must be positive")

        intact_sum = sum(length for length in core_recoveries if length >= 0.1)
        rqd = (intact_sum / total_length_m) * 100.0
        return float(min(100.0, max(0.0, rqd)))

    def calculate_jn(self, num_joint_sets: int) -> float:
        if num_joint_sets < 1:
            raise ValueError(f"num_joint_sets ({num_joint_sets}) must be at least 1")

        if num_joint_sets == 1:
            return 2.0
        elif num_joint_sets == 2:
            return 4.0
        elif num_joint_sets == 3:
            return 9.0
        elif num_joint_sets == 4:
            return 15.0
        else:
            return 20.0

    def calculate_jr(self, joint_type: str) -> float:
        jt = joint_type.lower().replace(" ", "_")
        mapping = {
            "discontinuous": 4.0,
            "rough_undulating": 3.0,
            "smooth_undulating": 2.0,
            "slickensided_undulating": 1.5,
            "rough_planar": 1.5,
            "smooth_planar": 1.0,
            "slickensided_planar": 0.5,
            "filled_planar": 1.0,
        }
        if jt not in mapping:
            return 1.5
        return mapping[jt]

    def calculate_ja(self, alteration: str) -> float:
        alt = alteration.lower().replace(" ", "_")
        mapping = {
            "tight_unaltered": 0.75,
            "unaltered": 0.75,
            "slightly_altered": 2.0,
            "clay_coated": 4.0,
            "clay_filled_thin": 8.0,
            "clay_filled_thick": 12.0,
            "crushed_rock": 20.0,
        }
        if alt not in mapping:
            return 2.0
        return mapping[alt]

    def calculate_jw(self, water_condition: str) -> float:
        wc = water_condition.lower().replace(" ", "_")
        mapping = {
            "dry": 1.0,
            "damp": 0.66,
            "wet": 0.5,
            "dripping": 0.33,
            "flowing": 0.15,
        }
        if wc not in mapping:
            return 1.0
        return mapping[wc]

    def calculate_srf(self, stress_condition: str) -> float:
        sc = stress_condition.lower().replace(" ", "_")
        mapping = {
            "favorable_low_stress": 1.0,
            "medium_stress": 1.0,
            "high_stress": 2.5,
            "mild_rockburst": 5.0,
            "heavy_rockburst": 10.0,
            "squeezing_rock": 15.0,
        }
        if sc not in mapping:
            return 1.0
        return mapping[sc]

    def calculate_esr(self, excavation_type: str) -> float:
        et = excavation_type.lower().replace(" ", "_")
        if "temporary" in et:
            return 3.0
        elif "permanent" in et:
            return 1.6
        elif "storage" in et or "power" in et:
            return 1.3
        else:
            return 1.0

    def calculate_q(
        self,
        rqd_pct: float,
        num_joint_sets: int,
        joint_type: str,
        alteration: str,
        water_condition: str,
        stress_condition: str,
        span_m: float = 10.0,
        excavation_type: str = "permanent_mine",
    ) -> Dict[str, Any]:
        jn = self.calculate_jn(num_joint_sets)
        jr = self.calculate_jr(joint_type)
        ja = self.calculate_ja(alteration)
        jw = self.calculate_jw(water_condition)
        srf = self.calculate_srf(stress_condition)
        esr = self.calculate_esr(excavation_type)

        q_val = float((rqd_pct / jn) * (jr / ja) * (jw / srf))
        de = span_m / esr

        if q_val > 400:
            roman = "I"
            desc = "Exceptionally Good"
            support = "Unsupported"
        elif q_val > 100:
            roman = "II"
            desc = "Extremely Good"
            support = "Spot bolting"
        elif q_val > 40:
            roman = "III"
            desc = "Very Good"
            support = "Spot bolting"
        elif q_val > 10:
            roman = "IV"
            desc = "Good"
            support = "Systematic bolting"
        elif q_val > 4:
            roman = "V"
            desc = "Fair"
            support = "Systematic bolting with fiber shotcrete"
        elif q_val > 1:
            roman = "VI"
            desc = "Poor"
            support = "Systematic bolting with 50-100mm shotcrete"
        elif q_val > 0.1:
            roman = "VII"
            desc = "Very Poor"
            support = "Fiber shotcrete and reinforced ribs"
        elif q_val > 0.01:
            roman = "VIII"
            desc = "Extremely Poor"
            support = "Heavy steel ribs and thick shotcrete"
        else:
            roman = "IX"
            desc = "Exceptionally Poor"
            support = "Cast concrete lining"

        return {
            "q_value": round(q_val, 3),
            "class_roman": roman,
            "description": desc,
            "equivalent_dimension_m": round(de, 2),
            "support_recommendation": support,
            "components": {
                "rqd": rqd_pct,
                "jn": jn,
                "jr": jr,
                "ja": ja,
                "jw": jw,
                "srf": srf,
            },
        }


def calculate_q_system(
    rqd_pct: float,
    jn: float,
    jr: float,
    ja: float,
    jw: float,
    srf: float,
) -> Dict[str, Any]:
    calc = QSystemCalculator()
    q_val = float((rqd_pct / jn) * (jr / ja) * (jw / srf))

    if q_val > 400:
        desc = "Exceptionally Good"
        support = "Unsupported"
    elif q_val > 100:
        desc = "Extremely Good"
        support = "Spot bolting"
    elif q_val > 40:
        desc = "Very Good"
        support = "Spot bolting"
    elif q_val > 10:
        desc = "Good"
        support = "Systematic bolting"
    elif q_val > 4:
        desc = "Fair"
        support = "Systematic bolting with fiber shotcrete"
    elif q_val > 1:
        desc = "Poor"
        support = "Systematic bolting with shotcrete"
    else:
        desc = "Very/Extremely Poor"
        support = "Steel ribs and thick shotcrete"

    return {
        "q_value": round(q_val, 3),
        "q_class": desc,
        "block_size_index": round(rqd_pct / jn, 2),
        "shear_strength_index": round(jr / ja, 2),
        "active_stress_index": round(jw / srf, 2),
        "support_recommendation": support,
    }
