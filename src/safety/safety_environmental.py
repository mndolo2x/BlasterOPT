"""
Streamlit interface for Safety & Environmental Assessment.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.safety.dust import DustModel
from src.safety.toxic_gases import GasModel
from src.safety.noise_prediction import NoiseModel
from src.safety.environmental_impact import EnvironmentalAssessment
from src.safety.risk_analysis import BlastRiskAnalyzer
from src.safety.renderer import (
    plot_dust_dispersion,
    plot_gas_concentration,
    plot_noise_attenuation,
    plot_risk_matrix,
    plot_environmental_breakdown,
)


def render_safety_environmental_page() -> None:
    st.title("🛡️ Safety & Environmental Assessment")
    st.caption("Comprehensive Dust, Toxic Gas, Noise, EIA, and ISO 31000 Risk Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💨 Dust Modeling",
        "💨 Toxic Gas",
        "🔊 Noise Prediction",
        "🌱 Environmental Impact",
        "📊 Risk Analysis",
    ])

    # ---------------------------------------------------------
    # TAB 1: DUST MODELING
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Dust Generation & Plume Dispersion (Tyupin & Bolotova 2026)")
        col1, col2 = st.columns(2)
        with col1:
            charge_mass = st.number_input("Total Explosive Charge Mass (kg)", value=1000.0, step=100.0)
            exp_type = st.selectbox("Explosive Type", ["ANFO", "Heavy ANFO", "Emulsion"], key="dust_exp")
        with col2:
            wind_speed = st.number_input("Wind Speed (m/s)", value=3.0, step=0.5)
            receptor_dist = st.number_input("Target Distance to Receptor (m)", value=500.0, step=50.0)

        dust_model = DustModel(explosive_type=exp_type, charge_mass_kg=charge_mass)
        dust_mass_res = dust_model.calculate_dust_mass(charge_mass)
        cloud_res = dust_model.calculate_dust_cloud_radius(charge_mass, wind_speed)
        pm10_res = dust_model.calculate_pm10_concentration(dust_mass_res["dust_mass_kg"], receptor_dist, wind_speed)

        st.markdown("### Numeric Results")
        res_df = pd.DataFrame([
            {"Parameter": "Total Dust Mass Generated", "Value": f"{dust_mass_res['dust_mass_kg']:.1f} kg"},
            {"Parameter": "Estimated Cloud Radius", "Value": f"{cloud_res['radius_m']:.1f} m"},
            {"Parameter": "PM10 Concentration at Receptor", "Value": f"{pm10_res['pm10_concentration_mg_m3']:.4f} mg/m³"},
            {"Parameter": "WHO 24-hr PM10 Limit", "Value": "0.0500 mg/m³"},
        ])
        st.table(res_df)

        if pm10_res["exceeds_limit"]:
            st.error("⚠️ WARNING: PM10 concentration exceeds WHO threshold (0.05 mg/m³)! Implement water curtain suppression.")
        else:
            st.success("✅ PM10 concentration is within safe WHO guidelines.")

        st.markdown("### Plain-Language Interpretation")
        st.info(f"Detonating {charge_mass:.0f} kg of {exp_type} under a wind speed of {wind_speed:.1f} m/s generates an estimated {dust_mass_res['dust_mass_kg']:.1f} kg of airborne dust particles. At a downwind distance of {receptor_dist:.0f} m, the resulting PM10 concentration is {pm10_res['pm10_concentration_mg_m3']:.4f} mg/m³.")

        dists = np.linspace(100, max(1000.0, receptor_dist * 1.5), 50)
        fig_dust = plot_dust_dispersion(dust_model, dists)
        st.plotly_chart(fig_dust, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: TOXIC GAS
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Toxic Gas Emissions & Re-Entry Time (Suceska et al. 2021)")
        col1, col2 = st.columns(2)
        with col1:
            gas_exp_mass = st.number_input("Explosive Mass (kg)", value=1000.0, step=100.0, key="gas_mass")
            gas_exp_type = st.selectbox("Explosive Type", ["ANFO", "Heavy ANFO", "Emulsion"], key="gas_exp")
        with col2:
            vent_rate = st.number_input("Ventilation Rate (m³/s)", value=10.0, step=1.0)
            tunnel_vol = st.number_input("Confined Volume (m³)", value=5000.0, step=500.0)

        gas_model = GasModel(explosive_type=gas_exp_type)
        emissions = gas_model.calculate_gas_emissions(gas_exp_mass)
        co_kg = emissions["CO_kg"]
        g_conc = gas_model.calculate_gas_concentration(co_kg, "CO", vent_rate, tunnel_vol, time_s=300.0)
        reentry_s = gas_model.calculate_reentry_time(co_kg, "CO", vent_rate, tunnel_vol)

        st.markdown("### Gas Emissions Breakdown")
        gas_df = pd.DataFrame([
            {"Gas": "Carbon Monoxide (CO)", "Emissions (kg)": emissions["CO_kg"], "Exposure Limit": "50 ppm (8-hr TWA)"},
            {"Gas": "Nitrogen Oxides (NOx)", "Emissions (kg)": emissions["NOx_kg"], "Exposure Limit": "3 ppm (8-hr TWA)"},
            {"Gas": "Carbon Dioxide (CO2)", "Emissions (kg)": emissions["CO2_kg"], "Exposure Limit": "5000 ppm"},
        ])
        st.table(gas_df)

        st.write(f"**Safe Re-entry Time (CO < 50 ppm):** `{reentry_s / 60.0:.1f} minutes` (`{reentry_s:.0f} seconds`)")

        if g_conc["exceeds_limit"]:
            st.error(f"⚠️ WARNING: Gas concentration ({g_conc['concentration_ppm']:.1f} ppm) exceeds safe exposure limits at 5 minutes!")
        else:
            st.success("✅ Gas concentration is within acceptable occupational safety thresholds.")

        st.markdown("### Plain-Language Interpretation")
        st.info(f"Firing {gas_exp_mass:.0f} kg of {gas_exp_type} produces {emissions['CO_kg']:.2f} kg of CO gas and {emissions['NOx_kg']:.2f} kg of NOx gas. Under a ventilation flushing rate of {vent_rate:.1f} m³/s in a {tunnel_vol:.0f} m³ space, personnel must delay re-entry for at least {reentry_s / 60.0:.1f} minutes.")

        times = np.linspace(30, 1800, 50)
        fig_gas = plot_gas_concentration(gas_model, times)
        st.plotly_chart(fig_gas, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: NOISE PREDICTION
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Airblast Noise Overpressure Prediction (Linehan & Wiss 1980)")
        col1, col2 = st.columns(2)
        with col1:
            q_delay = st.number_input("Max Charge Weight per Delay (kg)", value=150.0, step=10.0)
            burial_depth = st.number_input("Scaled Stemming / Burial Depth (m)", value=3.0, step=0.5)
        with col2:
            noise_dist = st.number_input("Distance to Receptor (m)", value=500.0, step=50.0, key="noise_dist")

        noise_model = NoiseModel(charge_per_delay_kg=q_delay, depth_of_burial_m=burial_depth)
        peak_res = noise_model.calculate_peak_overpressure(noise_dist)
        noise_res = noise_model.calculate_noise_at_distance(noise_dist)
        zone_res = noise_model.calculate_zone_of_impact(limit_db=120.0)

        st.markdown("### Numeric Results")
        noise_df = pd.DataFrame([
            {"Parameter": "Peak Overpressure", "Value": f"{peak_res['overpressure_kpa']:.4f} kPa"},
            {"Parameter": "Sound Pressure Level (SPL)", "Value": f"{noise_res['spl_db']:.1f} dBL"},
            {"Parameter": "Zone of Impact Distance (to 120 dBL)", "Value": f"{zone_res['distance_to_limit_m']:.1f} m"},
            {"Parameter": "Regulatory Airblast Limit", "Value": "120.0 dBL"},
        ])
        st.table(noise_df)

        if noise_res["exceeds_limit"]:
            st.error(f"⚠️ WARNING: Noise level ({noise_res['spl_db']:.1f} dBL) exceeds the Botswana statutory limit of 120 dBL!")
        else:
            st.success("✅ Noise overpressure is below the 120 dBL regulatory threshold.")

        st.markdown("### Plain-Language Interpretation")
        st.info(f"With a charge weight of {q_delay:.0f} kg per delay and {burial_depth:.1f} m burial depth, the calculated peak overpressure at {noise_dist:.0f} m is {peak_res['overpressure_kpa']:.4f} kPa ({noise_res['spl_db']:.1f} dBL). Noise levels remain above the 120 dBL regulatory limit out to a distance of {zone_res['distance_to_limit_m']:.1f} m.")

        dists_noise = np.linspace(100, max(1000.0, noise_dist * 1.5), 50)
        fig_noise = plot_noise_attenuation(noise_model, dists_noise)
        st.plotly_chart(fig_noise, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: ENVIRONMENTAL IMPACT
    # ---------------------------------------------------------
    with tab4:
        st.subheader("5-Category Environmental Impact Assessment (ISO 14001)")
        st.markdown("#### Receptor Distances (m)")
        c1, c2, c3 = st.columns(3)
        with c1:
            d_village = st.number_input("Village Distance (m)", value=800.0, step=50.0)
        with c2:
            d_road = st.number_input("Public Road Distance (m)", value=450.0, step=50.0)
        with c3:
            d_camp = st.number_input("Mine Camp Distance (m)", value=1200.0, step=50.0)

        receptors = {"village": d_village, "road": d_road, "camp": d_camp}
        blast_params = {
            "explosive_mass_kg": charge_mass,
            "max_charge_per_delay_kg": q_delay,
            "depth_of_burial_m": burial_depth,
            "burden_m": 3.5,
            "hole_diameter_m": 0.165,
        }

        eia = EnvironmentalAssessment(blast_params=blast_params, receptor_distances=receptors)
        total_eia = eia.calculate_total_impact()

        st.markdown(f"### Total EIA Score: `{total_eia['total_score']} / 25` — Rating: `{total_eia['overall_rating'].upper()}`")

        fig_radar = plot_environmental_breakdown(eia)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### Category Breakdown (1 = Negligible, 5 = Severe)")
        st.json(total_eia["breakdown"])

        st.markdown("### Priority Mitigation Measures")
        for mit in total_eia["priority_mitigations"]:
            st.write(f"- {mit}")

        if st.button("📄 Generate & Download EIA Report"):
            report_path = eia.generate_eia_report()
            with open(report_path, "r", encoding="utf-8") as f:
                st.download_button("Download Report Text", f.read(), file_name="EIA_Report.txt")

    # ---------------------------------------------------------
    # TAB 5: RISK ANALYSIS
    # ---------------------------------------------------------
    with tab5:
        st.subheader("8-Category ISO 31000 Blasting Risk Analysis")
        risk_analyzer = BlastRiskAnalyzer()

        risk_params = {
            "burden_m": 3.5,
            "hole_diameter_m": 0.165,
            "stemming_m": 2.5,
            "has_free_face": True,
            "is_wet": False,
            "explosive_type": exp_type,
            "closest_receptor_dist_m": min(d_village, d_road, d_camp),
            "max_charge_per_delay_kg": q_delay,
            "explosive_mass_kg": charge_mass,
            "depth_of_burial_m": burial_depth,
        }

        overall_risk = risk_analyzer.calculate_overall_risk(risk_params)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Overall Risk Level", overall_risk["overall_risk_level"].upper())
        with col_b:
            st.metric("Highest Risk Category", overall_risk["highest_risk_category"].capitalize())

        fig_matrix = plot_risk_matrix(risk_analyzer)
        st.plotly_chart(fig_matrix, use_container_width=True)

        if overall_risk["critical_items"]:
            st.error(f"⚠️ CRITICAL/HIGH RISKS IDENTIFIED: {', '.join(overall_risk['critical_items'])}")

        st.markdown("### Priority Action Plan")
        for act in overall_risk["priority_mitigations"]:
            st.write(f"- {act}")
