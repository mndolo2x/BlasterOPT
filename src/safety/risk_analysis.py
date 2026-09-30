"""
Comprehensive quantitative risk matrix per ISO 31000 / AS/NZS 4360 risk management standard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from src.safety.dust import DustModel
from src.safety.toxic_gases import GasModel
from src.safety.noise_prediction import NoiseModel


class BlastRiskAnalyzer:
    """
    Comprehensive risk analysis for blasting operations.

    Evaluates 8 risk categories:
        1. Flyrock (safety critical)
        2. Ground vibration (structural damage)
        3. Airblast (community annoyance)
        4. Noise (health)
        5. Dust (health)
        6. Toxic gas (health)
        7. Misfire (safety critical)
        8. Slope instability (geotechnical)

    Uses a risk matrix approach:
        Risk = Likelihood × Consequence

    Likelihood: 1-5 (1 = rare, 5 = almost certain)
    Consequence: 1-5 (1 = negligible, 5 = catastrophic)
    """

    RISK_LEVELS = {
        (1, 1): "low", (1, 2): "low", (1, 3): "moderate", (1, 4): "moderate", (1, 5): "high",
        (2, 1): "low", (2, 2): "moderate", (2, 3): "moderate", (2, 4): "high", (2, 5): "high",
        (3, 1): "moderate", (3, 2): "moderate", (3, 3): "high", (3, 4): "high", (3, 5): "critical",
        (4, 1): "moderate", (4, 2): "high", (4, 3): "high", (4, 4): "critical", (4, 5): "critical",
        (5, 1): "high", (5, 2): "high", (5, 3): "critical", (5, 4): "critical", (5, 5): "critical",
    }

    def _get_level(self, likelihood: int, consequence: int) -> str:
        l_clamped = max(1, min(5, likelihood))
        c_clamped = max(1, min(5, consequence))
        return self.RISK_LEVELS.get((l_clamped, c_clamped), "moderate")

    def assess_flyrock_risk(self, blast_params: dict) -> dict:
        """
        Flyrock risk assessment.

        Likelihood factors:
        - Burden < 2× hole diameter: +2
        - Stemming < 0.5× burden: +2
        - No free face: +3
        - Wet holes with ANFO: +2

        Consequence factors:
        - Receptor within 500 m: +3
        - Receptor within 1000 m: +2
        - No flyrock protection: +2
        """
        burden_m = float(blast_params.get("burden_m", 3.5))
        hole_diam_m = float(blast_params.get("hole_diameter_m", 0.165))
        stemming_m = float(blast_params.get("stemming_m", 2.5))
        has_free_face = bool(blast_params.get("has_free_face", True))
        is_wet = bool(blast_params.get("is_wet", False))
        explosive_type = str(blast_params.get("explosive_type", "ANFO")).upper()
        receptor_dist_m = float(blast_params.get("closest_receptor_dist_m", 600.0))
        has_protection = bool(blast_params.get("has_flyrock_protection", False))

        l_points = 1
        if burden_m < (2.0 * hole_diam_m):
            l_points += 2
        if stemming_m < (0.5 * burden_m):
            l_points += 2
        if not has_free_face:
            l_points += 3
        if is_wet and ("ANFO" in explosive_type):
            l_points += 2

        likelihood = max(1, min(5, l_points))

        c_points = 1
        if receptor_dist_m < 500.0:
            c_points += 3
        elif receptor_dist_m < 1000.0:
            c_points += 2
        if not has_protection:
            c_points += 2

        consequence = max(1, min(5, c_points))
        score = likelihood * consequence
        level = self._get_level(likelihood, consequence)

        mitigations = []
        if likelihood >= 3:
            mitigations.append("Audit face burden and stemming height using 3D laser profiling")
        if consequence >= 3:
            mitigations.append("Clear exclusion zone up to 500m radius and deploy blast guards")

        return {
            "likelihood": likelihood,
            "consequence": consequence,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": mitigations,
        }

    def assess_vibration_risk(self, blast_params: dict) -> dict:
        q_max = float(blast_params.get("max_charge_per_delay_kg", 150.0))
        dist_m = float(blast_params.get("closest_receptor_dist_m", 500.0))
        k_vib = float(blast_params.get("k_vib", 1140.0))
        b_vib = float(blast_params.get("b_vib", 1.6))

        sd = dist_m / (q_max ** 0.5) if q_max > 0 else 100.0
        ppv_mms = k_vib * (sd ** (-b_vib))

        likelihood = 1
        if ppv_mms > 25.0:
            likelihood = 5
        elif ppv_mms > 15.0:
            likelihood = 4
        elif ppv_mms > 10.0:
            likelihood = 3
        elif ppv_mms > 5.0:
            likelihood = 2

        consequence = 3 if dist_m < 500.0 else 2
        score = likelihood * consequence
        level = self._get_level(likelihood, consequence)

        mitigations = []
        if likelihood >= 3:
            mitigations.append("Reduce Q_max by deck charging and optimize delay timing interval")

        return {
            "likelihood": likelihood,
            "consequence": consequence,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": mitigations,
        }

    def assess_airblast_risk(self, blast_params: dict) -> dict:
        q_max = float(blast_params.get("max_charge_per_delay_kg", 150.0))
        dist_m = float(blast_params.get("closest_receptor_dist_m", 500.0))
        depth_burial = float(blast_params.get("depth_of_burial_m", 3.0))

        noise_model = NoiseModel(charge_per_delay_kg=q_max, depth_of_burial_m=depth_burial)
        n_res = noise_model.calculate_noise_at_distance(dist_m)
        spl_db = n_res["spl_db"]

        likelihood = 1
        if spl_db > 130.0:
            likelihood = 5
        elif spl_db > 120.0:
            likelihood = 4
        elif spl_db > 115.0:
            likelihood = 3
        elif spl_db > 105.0:
            likelihood = 2

        consequence = 3 if dist_m < 500.0 else 2
        score = likelihood * consequence
        level = self._get_level(likelihood, consequence)

        mitigations = []
        if likelihood >= 3:
            mitigations.append("Cover surface initiation detonating cord with adequate crushed rock stemming")

        return {
            "likelihood": likelihood,
            "consequence": consequence,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": mitigations,
        }

    def assess_dust_risk(self, blast_params: dict) -> dict:
        q_total = float(blast_params.get("explosive_mass_kg", 1000.0))
        dist_m = float(blast_params.get("closest_receptor_dist_m", 500.0))
        wind_speed = float(blast_params.get("wind_speed_m_s", 3.0))

        dust_model = DustModel()
        d_mass = dust_model.calculate_dust_mass(q_total)
        p_res = dust_model.calculate_pm10_concentration(d_mass["dust_mass_kg"], dist_m, wind_speed)
        pm10_mg = p_res["pm10_concentration_mg_m3"]

        likelihood = 1
        if pm10_mg > 0.20:
            likelihood = 5
        elif pm10_mg > 0.10:
            likelihood = 4
        elif pm10_mg > 0.05:
            likelihood = 3
        elif pm10_mg > 0.02:
            likelihood = 2

        consequence = 2
        score = likelihood * consequence
        level = self._get_level(likelihood, consequence)

        mitigations = []
        if likelihood >= 3:
            mitigations.append("Apply water spraying on muckpile and bench before firing")

        return {
            "likelihood": likelihood,
            "consequence": consequence,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": mitigations,
        }

    def assess_gas_risk(self, blast_params: dict) -> dict:
        q_total = float(blast_params.get("explosive_mass_kg", 1000.0))
        exp_type = str(blast_params.get("explosive_type", "ANFO"))

        gas_model = GasModel(explosive_type=exp_type)
        emissions = gas_model.calculate_gas_emissions(q_total)
        co_kg = emissions["CO_kg"]
        g_conc = gas_model.calculate_gas_concentration(co_kg, "CO", 10.0, 5000.0)
        co_ppm = g_conc["concentration_ppm"]

        likelihood = 1
        if co_ppm > 100.0:
            likelihood = 5
        elif co_ppm > 50.0:
            likelihood = 4
        elif co_ppm > 25.0:
            likelihood = 3
        elif co_ppm > 10.0:
            likelihood = 2

        consequence = 3
        score = likelihood * consequence
        level = self._get_level(likelihood, consequence)

        mitigations = []
        if likelihood >= 3:
            mitigations.append("Enforce 30-minute re-entry delay and test post-blast gas concentrations")

        return {
            "likelihood": likelihood,
            "consequence": consequence,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": mitigations,
        }

    def assess_misfire_risk(self, blast_params: dict) -> dict:
        is_wet = bool(blast_params.get("is_wet", False))
        exp_type = str(blast_params.get("explosive_type", "ANFO")).upper()
        detonator_type = str(blast_params.get("detonator_type", "pyrotechnic")).lower()

        likelihood = 1
        if is_wet and ("ANFO" in exp_type):
            likelihood += 2
        if "pyrotechnic" in detonator_type:
            likelihood += 1

        consequence = 4  # Safety critical
        likelihood = max(1, min(5, likelihood))
        score = likelihood * consequence
        level = self._get_level(likelihood, consequence)

        mitigations = []
        if likelihood >= 3:
            mitigations.append("Switch to water-resistant emulsion explosive and electronic detonators")

        return {
            "likelihood": likelihood,
            "consequence": consequence,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": mitigations,
        }

    def assess_noise_risk(self, blast_params: dict) -> dict:
        res = self.assess_airblast_risk(blast_params)
        res["mitigation_measures"] = ["Provide personal hearing protection equipment to mine personnel"]
        return res

    def assess_slope_instability_risk(self, blast_params: dict) -> dict:
        ppv_risk = self.assess_vibration_risk(blast_params)
        l = ppv_risk["likelihood"]
        c = 3
        score = l * c
        level = self._get_level(l, c)
        return {
            "likelihood": l,
            "consequence": c,
            "risk_level": level,
            "risk_score": score,
            "mitigation_measures": ["Perform pre-split trim blasting along pit walls"],
        }

    def calculate_overall_risk(self, blast_params: dict) -> dict:
        """
        Aggregates all risk assessments.

        Returns:
            {
                "overall_risk_level": str,
                "highest_risk_category": str,
                "risk_matrix": pd.DataFrame,  # 5x5 matrix
                "critical_items": list[str],
                "priority_mitigations": list[str],
            }
        """
        assessments = {
            "flyrock": self.assess_flyrock_risk(blast_params),
            "vibration": self.assess_vibration_risk(blast_params),
            "airblast": self.assess_airblast_risk(blast_params),
            "noise": self.assess_noise_risk(blast_params),
            "dust": self.assess_dust_risk(blast_params),
            "gas": self.assess_gas_risk(blast_params),
            "misfire": self.assess_misfire_risk(blast_params),
            "slope_instability": self.assess_slope_instability_risk(blast_params),
        }

        # Find highest score and category
        highest_cat = max(assessments.keys(), key=lambda k: assessments[k]["risk_score"])
        highest_score = assessments[highest_cat]["risk_score"]
        overall_level = assessments[highest_cat]["risk_level"]

        # Build 5x5 matrix counts
        matrix_grid = [[0 for _ in range(5)] for _ in range(5)]
        critical_items = []
        priority_mitigations = []

        for cat, a in assessments.items():
            l = a["likelihood"] - 1
            c = a["consequence"] - 1
            matrix_grid[l][c] += 1

            if a["risk_level"] in ["high", "critical"]:
                critical_items.append(f"{cat.capitalize()} ({a['risk_level'].upper()})")
            priority_mitigations.extend(a["mitigation_measures"])

        df_matrix = pd.DataFrame(
            matrix_grid,
            index=[f"L{i}" for i in range(1, 6)],
            columns=[f"C{i}" for i in range(1, 6)],
        )

        priority_mitigations = list(dict.fromkeys(priority_mitigations))

        return {
            "overall_risk_level": overall_level,
            "highest_risk_category": highest_cat,
            "risk_matrix": df_matrix,
            "critical_items": critical_items,
            "priority_mitigations": priority_mitigations,
            "assessments": assessments,
        }

    def generate_risk_matrix_plot(self, blast_params: dict) -> go.Figure:
        """
        Creates a 5x5 risk matrix heatmap with all risks plotted.
        """
        res = self.calculate_overall_risk(blast_params)
        assessments = res["assessments"]

        # Base 5x5 risk severity color matrix
        z_color = [
            [1, 1, 2, 2, 3],
            [1, 2, 2, 3, 3],
            [2, 2, 3, 3, 4],
            [2, 3, 3, 4, 4],
            [3, 3, 4, 4, 4],
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=z_color,
                x=["C1 (Negligible)", "C2 (Minor)", "C3 (Moderate)", "C4 (Major)", "C5 (Catastrophic)"],
                y=["L1 (Rare)", "L2 (Unlikely)", "L3 (Possible)", "L4 (Likely)", "L5 (Almost Certain)"],
                colorscale=[[0, "green"], [0.33, "yellow"], [0.66, "orange"], [1.0, "red"]],
                showscale=False,
                opacity=0.6,
            )
        )

        # Plot points for each risk
        for cat, a in assessments.items():
            fig.add_trace(
                go.Scatter(
                    x=[a["consequence"] - 1],
                    y=[a["likelihood"] - 1],
                    mode="markers+text",
                    name=cat.capitalize(),
                    text=[cat.capitalize()],
                    textposition="top center",
                    marker=dict(size=14, line=dict(width=2, color="DarkSlateGrey")),
                )
            )

        fig.update_layout(
            title="<b>5x5 Blasting Risk Assessment Matrix (ISO 31000)</b>",
            xaxis_title="Consequence Severity",
            yaxis_title="Likelihood Probability",
            template="plotly_white",
        )
        return fig


def calculate_risk_matrix(
    ppv_mms: float,
    airblast_dbl: float,
    flyrock_m: float,
    hole_collisions_count: int,
    backbreak_m: float,
) -> Dict[str, Any]:
    """
    Calculate ISO 31000 quantitative risk matrix score and hazard ranking for blast design.

    Returns:
        {
            "overall_risk_score": float,  # 1 to 25
            "risk_level": str,  # 'low', 'medium', 'high', 'extreme'
            "hazards": list[dict],
            "action_plan": list[str],
        }
    """
    hazards: List[Dict[str, Any]] = []
    actions: List[str] = []

    # 1. Vibration
    if ppv_mms > 10.0:
        hazards.append({"hazard": "Ground Vibration Exceedance", "severity": 5, "likelihood": 4, "score": 20, "level": "high"})
        actions.append("Reduce max charge per delay or increase monitoring distance.")

    # 2. Flyrock
    if flyrock_m > 250.0:
        hazards.append({"hazard": "Flyrock Exclusion Zone Breach", "severity": 5, "likelihood": 5, "score": 25, "level": "extreme"})
        actions.append("Increase stemming height and check front row burden.")

    # 3. Collision
    if hole_collisions_count > 0:
        hazards.append({"hazard": "Hole-to-Hole Toe Collision", "severity": 5, "likelihood": 4, "score": 20, "level": "high"})
        actions.append("Re-design collar spacing or adjust drill hole dip angles.")

    # 4. Airblast
    if airblast_dbl > 120.0:
        hazards.append({"hazard": "Airblast Overpressure Exceedance", "severity": 4, "likelihood": 4, "score": 16, "level": "medium"})
        actions.append("Cover surface detonation trunklines and avoid blasting under temperature inversions.")

    max_score = max([h["score"] for h in hazards], default=4)

    if max_score >= 20:
        risk_level = "extreme"
    elif max_score >= 15:
        risk_level = "high"
    elif max_score >= 8:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "overall_risk_score": max_score,
        "risk_level": risk_level,
        "hazards": hazards,
        "action_plan": actions,
    }
