"""
Safety, Environmental & Risk Analysis Module for BlastOpt Botswana.

Implements physics and empirical models for:
1. Noise prediction and distance attenuation (Linehan & Wiss 1980 USBM / EPA WA 2011).
2. Dust dispersion and emission modeling (Tyupin & Bolotova 2026 / Pasquill-Gifford).
3. Toxic gas emissions (CO, NOx, CO2 per Suceska et al. 2021 & Newgold factors).
4. Environmental Impact Assessment (EIA) scoring across 5 categories.
5. 5x5 Likelihood x Consequence Risk Matrix and Blast Risk Analysis.
6. Plotly visualizations and Streamlit UI page renderer.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional


class NoiseModel:
    """
    Blast noise prediction and distance attenuation model.
    Based on USBM RI 8507 / Linehan & Wiss (1980) and EPA WA (2011) standards.
    """

    @staticmethod
    def predict_noise_level(
        charge_mass_per_delay_kg: float,
        distance_m: float,
        temperature_c: float = 25.0,
        wind_speed_m_s: float = 3.0,
        humidity_pct: float = 40.0,
    ) -> Dict[str, float]:
        """
        Predicts peak sound pressure level (dBA and dBL) at distance_m.
        """
        W = max(charge_mass_per_delay_kg, 0.1)
        D = max(distance_m, 10.0)

        # Scaled distance for airblast/noise
        scaled_distance = D / (W ** (1.0 / 3.0))

        # Peak linear overpressure dBL (Siskind et al. 1980)
        dbl = 165.0 - 24.0 * np.log10(scaled_distance)

        # Environmental attenuation adjustments for dBA conversion
        # Atmospheric absorption ~ 0.5 dBA per 100m at 25C, 40% humidity
        atmos_attenuation = (D / 100.0) * 0.5

        # Wind effect factor
        wind_factor = 1.5 if wind_speed_m_s > 5.0 else 0.0

        # Estimated A-weighted noise dBA
        dba = dbl - 26.0 - atmos_attenuation + wind_factor

        return {
            "dbl_peak": float(np.clip(dbl, 40.0, 140.0)),
            "dba_peak": float(np.clip(dba, 30.0, 120.0)),
            "scaled_distance": float(scaled_distance),
            "atmos_attenuation_dba": float(atmos_attenuation),
        }

    @staticmethod
    def calculate_noise_attenuation_curve(
        charge_mass_per_delay_kg: float,
        max_distance_m: float = 2000.0,
        num_points: int = 50,
    ) -> pd.DataFrame:
        """Generates distance vs noise level dataframe."""
        distances = np.linspace(50.0, max_distance_m, num_points)
        rows = []
        for d in distances:
            res = NoiseModel.predict_noise_level(charge_mass_per_delay_kg, d)
            rows.append({
                "distance_m": d,
                "dba_peak": res["dba_peak"],
                "dbl_peak": res["dbl_peak"],
            })
        return pd.DataFrame(rows)


class DustModel:
    """
    Dust dispersion and particulate emission model for open-pit blasting.
    Based on Tyupin & Bolotova (2026) and Pasquill-Gifford Gaussian dispersion.
    """

    @staticmethod
    def calculate_dust_emissions(
        total_explosive_mass_kg: float,
        rock_density_t_m3: float = 2.65,
        silt_content_pct: float = 12.0,
        moisture_content_pct: float = 3.0,
    ) -> Dict[str, float]:
        """
        Calculates total PM10 and PM2.5 dust emission mass (kg).
        """
        # Base dust emission factor per tonne of blasted rock
        # Higher silt content increases dust; higher moisture suppresses dust.
        moisture_factor = max(0.2, 1.0 - 0.15 * moisture_content_pct)
        silt_factor = silt_content_pct / 10.0

        # Empirical dust mass generated (kg per kg explosive)
        pm10_per_kg_exp = 0.45 * silt_factor * moisture_factor
        pm2_5_per_kg_exp = 0.08 * silt_factor * moisture_factor

        pm10_mass = total_explosive_mass_kg * pm10_per_kg_exp
        pm2_5_mass = total_explosive_mass_kg * pm2_5_per_kg_exp

        return {
            "pm10_mass_kg": float(pm10_mass),
            "pm2_5_mass_kg": float(pm2_5_mass),
            "total_dust_kg": float(pm10_mass + pm2_5_mass),
        }

    @staticmethod
    def calculate_dust_dispersion_profile(
        pm10_mass_kg: float,
        wind_speed_m_s: float = 3.0,
        max_distance_m: float = 3000.0,
        num_points: int = 50,
    ) -> pd.DataFrame:
        """Generates downwind PM10 concentration profile (mg/m3)."""
        distances = np.linspace(100.0, max_distance_m, num_points)
        rows = []
        u = max(wind_speed_m_s, 0.5)

        for d in distances:
            # Pasquill-Gifford dispersion parameter sigma_y, sigma_z approximations
            sigma_y = 0.11 * d * (1.0 + 0.0001 * d) ** -0.5
            sigma_z = 0.08 * d * (1.0 + 0.0002 * d) ** -0.5

            # Ground level peak concentration (mg/m3)
            conc_mg_m3 = (pm10_mass_kg * 1000.0) / (np.pi * u * sigma_y * sigma_z)
            rows.append({
                "distance_m": d,
                "pm10_concentration_mg_m3": float(np.clip(conc_mg_m3, 0.01, 50.0)),
            })
        return pd.DataFrame(rows)


class GasModel:
    """
    Toxic gas emissions (CO, NOx, CO2) prediction model.
    Based on Suceska et al. (2021) and Newgold sustainability emission factors.
    """

    @staticmethod
    def predict_toxic_gases(
        total_explosive_mass_kg: float,
        explosive_type: str = "ANFO",
        water_in_hole: bool = False,
    ) -> Dict[str, float]:
        """
        Predicts toxic gas volumes (liters and CO2 equivalent kg).
        """
        # Water in hole or improper oxygen balance increases NOx/CO generation
        water_factor = 2.2 if water_in_hole else 1.0

        if explosive_type.upper() == "EMULSION":
            co_factor = 12.0  # L/kg
            nox_factor = 4.5 * water_factor  # L/kg
            co2_factor = 0.18  # kg CO2/kg explosive
        else:  # ANFO
            co_factor = 18.0 * water_factor  # L/kg
            nox_factor = 6.0 * water_factor  # L/kg
            co2_factor = 0.22  # kg CO2/kg explosive

        co_volume_l = total_explosive_mass_kg * co_factor
        nox_volume_l = total_explosive_mass_kg * nox_factor
        co2_mass_kg = total_explosive_mass_kg * co2_factor

        return {
            "co_volume_liters": float(co_volume_l),
            "nox_volume_liters": float(nox_volume_l),
            "co2_emissions_kg": float(co2_mass_kg),
            "total_toxic_gas_liters": float(co_volume_l + nox_volume_l),
        }


class EnvironmentalAssessment:
    """
    Environmental Impact Assessment (EIA) rating across 5 core impact categories.
    Outputs normalized 1-5 rating scale (1 = minimal impact, 5 = severe impact).
    """

    @staticmethod
    def evaluate_impacts(
        ppv_mms: float,
        airblast_dbl: float,
        flyrock_m: float,
        dust_pm10_kg: float,
        co2_mass_kg: float,
    ) -> Dict[str, Any]:
        """Evaluates 1-5 scale scores per category."""
        # 1. Ground Vibration Impact Score
        score_vib = 1 if ppv_mms < 2.0 else (2 if ppv_mms < 5.0 else (3 if ppv_mms < 8.0 else (4 if ppv_mms < 10.0 else 5)))

        # 2. Airblast Noise Impact Score
        score_air = 1 if airblast_dbl < 105.0 else (2 if airblast_dbl < 112.0 else (3 if airblast_dbl < 118.0 else (4 if airblast_dbl < 120.0 else 5)))

        # 3. Flyrock Safety Hazard Score
        score_fly = 1 if flyrock_m < 50.0 else (2 if flyrock_m < 100.0 else (3 if flyrock_m < 180.0 else (4 if flyrock_m < 250.0 else 5)))

        # 4. Dust Dispersion Score
        score_dust = 1 if dust_pm10_kg < 20.0 else (2 if dust_pm10_kg < 50.0 else (3 if dust_pm10_kg < 120.0 else (4 if dust_pm10_kg < 250.0 else 5)))

        # 5. Carbon Footprint Score
        score_carbon = 1 if co2_mass_kg < 100.0 else (2 if co2_mass_kg < 300.0 else (3 if co2_mass_kg < 700.0 else (4 if co2_mass_kg < 1500.0 else 5)))

        overall_rating = float(np.mean([score_vib, score_air, score_fly, score_dust, score_carbon]))

        return {
            "vibration_score": score_vib,
            "airblast_score": score_air,
            "flyrock_score": score_fly,
            "dust_score": score_dust,
            "carbon_score": score_carbon,
            "overall_impact_rating": overall_rating,
            "is_compliant": bool(score_vib <= 4 and score_air <= 4 and score_fly <= 4),
        }


class BlastRiskAnalyzer:
    """
    Evaluates an 8-category risk matrix based on 5x5 Likelihood x Consequence methodology.
    """

    CATEGORIES = [
        "Ground Vibration (Pit Wall)",
        "Airblast Overpressure",
        "Flyrock Confinement",
        "Stemming Ejection",
        "Misfire / Cut-off",
        "Toxic Fumes Generation",
        "Dust Cloud Dispersion",
        "Crusher Jamming (Boulders)",
    ]

    @staticmethod
    def evaluate_risk_matrix(
        ppv_mms: float,
        airblast_dbl: float,
        flyrock_m: float,
        stemming_m: float,
        burden_m: float,
    ) -> pd.DataFrame:
        """Returns risk evaluation dataframe for all 8 categories."""
        rows = []

        # Risk 1: Ground Vibration
        l_vib = 4 if ppv_mms > 8.0 else (2 if ppv_mms > 4.0 else 1)
        c_vib = 4 if ppv_mms > 10.0 else 2
        rows.append({"Category": "Ground Vibration (Pit Wall)", "Likelihood": l_vib, "Consequence": c_vib, "Risk Score": l_vib * c_vib})

        # Risk 2: Airblast
        l_air = 4 if airblast_dbl > 118.0 else (2 if airblast_dbl > 110.0 else 1)
        c_air = 4 if airblast_dbl > 120.0 else 2
        rows.append({"Category": "Airblast Overpressure", "Likelihood": l_air, "Consequence": c_air, "Risk Score": l_air * c_air})

        # Risk 3: Flyrock
        l_fly = 4 if flyrock_m > 200.0 else (2 if flyrock_m > 100.0 else 1)
        c_fly = 5 if flyrock_m > 250.0 else 3
        rows.append({"Category": "Flyrock Confinement", "Likelihood": l_fly, "Consequence": c_fly, "Risk Score": l_fly * c_fly})

        # Risk 4: Stemming Ejection
        stem_ratio = stemming_m / max(burden_m, 0.1)
        l_stem = 4 if stem_ratio < 0.7 else (2 if stem_ratio < 0.9 else 1)
        c_stem = 3
        rows.append({"Category": "Stemming Ejection", "Likelihood": l_stem, "Consequence": c_stem, "Risk Score": l_stem * c_stem})

        # Risk 5: Misfire / Cut-off
        rows.append({"Category": "Misfire / Cut-off", "Likelihood": 1, "Consequence": 4, "Risk Score": 4})

        # Risk 6: Toxic Fumes
        rows.append({"Category": "Toxic Fumes Generation", "Likelihood": 2, "Consequence": 3, "Risk Score": 6})

        # Risk 7: Dust Cloud Dispersion
        rows.append({"Category": "Dust Cloud Dispersion", "Likelihood": 3, "Consequence": 2, "Risk Score": 6})

        # Risk 8: Crusher Jamming
        rows.append({"Category": "Crusher Jamming (Boulders)", "Likelihood": 2, "Consequence": 3, "Risk Score": 6})

        df_risk = pd.DataFrame(rows)

        def risk_level(score):
            if score >= 15:
                return "CRITICAL 🔴"
            elif score >= 10:
                return "HIGH 🟠"
            elif score >= 5:
                return "MEDIUM 🟡"
            else:
                return "LOW 🟢"

        df_risk["Risk Level"] = df_risk["Risk Score"].apply(risk_level)
        return df_risk


def render_safety_environmental_page(lang_code: str = "en"):
    """Renders the Safety, Noise & Environmental Assessment dashboard in Streamlit."""
    st.header("🛡️ Safety, Noise & Environmental Impact Assessment")
    st.markdown(
        "Real-time physics-guided evaluation of **noise attenuation**, **dust dispersion**, "
        "**toxic gas emissions**, **5-category EIA ratings**, and **5x5 risk matrix** for open-pit diamond mining operations."
    )

    tab_noise, tab_dust, tab_gas, tab_eia, tab_risk = st.tabs([
        "🔊 Noise Prediction",
        "🌪️ Dust Dispersion",
        "💨 Toxic Gas Emissions",
        "🌱 Environmental Assessment",
        "🎯 5x5 Risk Matrix",
    ])

    # Inputs sidebar / top controls
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        charge_delay = st.number_input("Max Charge per Delay (kg)", 10.0, 3000.0, 640.0, step=20.0)
        dist_m = st.number_input("Distance to Receptor / Boundary (m)", 50.0, 3000.0, 450.0, step=25.0)
    with col_in2:
        tot_exp_mass = st.number_input("Total Explosive Mass (kg)", 100.0, 50000.0, 15000.0, step=500.0)
        exp_type = st.selectbox("Explosive Product", ["ANFO", "Emulsion"])
    with col_in3:
        stemming_val = st.number_input("Stemming Length (m)", 1.0, 10.0, 5.0, step=0.2)
        burden_val = st.number_input("Burden (m)", 2.0, 12.0, 6.0, step=0.2)

    # 1. TAB NOISE
    with tab_noise:
        st.subheader("🔊 Noise Prediction & Distance Attenuation Curve")
        noise_res = NoiseModel.predict_noise_level(charge_delay, dist_m)

        n1, n2, n3 = st.columns(3)
        n1.metric("Predicted Peak Noise (dBA)", f"{noise_res['dba_peak']:.1f} dBA")
        n2.metric("Predicted Linear Overpressure (dBL)", f"{noise_res['dbl_peak']:.1f} dBL")
        n3.metric("Airblast Scaled Distance", f"{noise_res['scaled_distance']:.2f}")

        df_noise = NoiseModel.calculate_noise_attenuation_curve(charge_delay)
        fig_noise = px.line(
            df_noise,
            x="distance_m",
            y=["dba_peak", "dbl_peak"],
            labels={"value": "Decibels (dB)", "distance_m": "Distance from Blast (m)"},
            title="<b>Noise Level Attenuation vs Distance</b>",
            color_discrete_sequence=["#2962FF", "#D50000"],
        )
        st.plotly_chart(fig_noise, use_container_width=True)

    # 2. TAB DUST
    with tab_dust:
        st.subheader("🌪️ Dust Emission & Downwind Dispersion")
        dust_res = DustModel.calculate_dust_emissions(tot_exp_mass)

        d1, d2, d3 = st.columns(3)
        d1.metric("PM10 Dust Mass", f"{dust_res['pm10_mass_kg']:.1f} kg")
        d2.metric("PM2.5 Fine Dust Mass", f"{dust_res['pm2_5_mass_kg']:.1f} kg")
        d3.metric("Total Particulate Mass", f"{dust_res['total_dust_kg']:.1f} kg")

        df_dust = DustModel.calculate_dust_dispersion_profile(dust_res["pm10_mass_kg"])
        fig_dust = px.line(
            df_dust,
            x="distance_m",
            y="pm10_concentration_mg_m3",
            labels={"pm10_concentration_mg_m3": "PM10 Concentration (mg/m³)", "distance_m": "Downwind Distance (m)"},
            title="<b>Downwind PM10 Ground-Level Concentration Profile</b>",
            color_discrete_sequence=["#FF6D00"],
        )
        st.plotly_chart(fig_dust, use_container_width=True)

    # 3. TAB GAS
    with tab_gas:
        st.subheader("💨 Toxic Fume & Greenhouse Gas Emissions")
        gas_res = GasModel.predict_toxic_gases(tot_exp_mass, exp_type)

        g1, g2, g3 = st.columns(3)
        g1.metric("Carbon Monoxide (CO)", f"{gas_res['co_volume_liters']:.0f} L")
        g2.metric("Nitrogen Oxides (NOx)", f"{gas_res['nox_volume_liters']:.0f} L")
        g3.metric("CO2 Equivalent Mass", f"{gas_res['co2_emissions_kg']:.1f} kg CO2e")

    # 4. TAB EIA
    with tab_eia:
        st.subheader("🌱 Environmental Impact Assessment (EIA) Score")
        eia_res = EnvironmentalAssessment.evaluate_impacts(
            ppv_mms=8.5,
            airblast_dbl=noise_res["dbl_peak"],
            flyrock_m=120.0,
            dust_pm10_kg=dust_res["pm10_mass_kg"],
            co2_mass_kg=gas_res["co2_emissions_kg"],
        )

        st.metric("Overall Environmental Impact Rating (1-5)", f"{eia_res['overall_impact_rating']:.2f} / 5.0")

        df_eia = pd.DataFrame([
            {"Category": "Ground Vibration", "Rating (1-5)": eia_res["vibration_score"]},
            {"Category": "Airblast Overpressure", "Rating (1-5)": eia_res["airblast_score"]},
            {"Category": "Flyrock Safety", "Rating (1-5)": eia_res["flyrock_score"]},
            {"Category": "Dust Dispersion", "Rating (1-5)": eia_res["dust_score"]},
            {"Category": "Carbon Emissions", "Rating (1-5)": eia_res["carbon_score"]},
        ])
        fig_eia = px.bar(df_eia, x="Category", y="Rating (1-5)", title="<b>EIA Category Rating Profile</b>", color="Rating (1-5)")
        st.plotly_chart(fig_eia, use_container_width=True)

    # 5. TAB RISK
    with tab_risk:
        st.subheader("🎯 5x5 Likelihood x Consequence Risk Matrix")
        df_risk = BlastRiskAnalyzer.evaluate_risk_matrix(
            ppv_mms=8.5,
            airblast_dbl=noise_res["dbl_peak"],
            flyrock_m=120.0,
            stemming_m=stemming_val,
            burden_m=burden_val,
        )
        st.dataframe(df_risk, use_container_width=True)
