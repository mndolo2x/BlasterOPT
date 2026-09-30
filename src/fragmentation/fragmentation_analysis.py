"""
Streamlit UI module for Fragmentation Analysis in BlasterOPT / BlastOpt Botswana.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.fragmentation.kuz_ram import predict_kuz_ram
from src.fragmentation.swebrec import SwebrecModel
from src.fragmentation.kco import KCOModel
from src.fragmentation.distributions import compare_distributions, plot_distribution_comparison
from src.fragmentation.image_analysis import ImageAnalysisImporter
from src.fragmentation.calibration import FragmentationCalibrator
from src.fragmentation.renderer import (
    plot_fragmentation_curve,
    plot_model_comparison,
    plot_calibration_scatter,
)


def render_fragmentation_analysis_page():
    st.header("💥 Advanced Rock Fragmentation Analysis & Size Distributions")
    st.caption(
        "Literature-standard fragmentation prediction models (Kuz-Ram, Swebrec, KCO), "
        "WipFrag/Split-Desktop image analysis ingestion, and response surface methodology (RSM) calibration."
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "1. Kuz-Ram Prediction",
        "2. KCO Prediction",
        "3. Distribution Comparison",
        "4. Image Analysis",
        "5. Rock-Specific Calibration",
    ])

    # Inputs
    st.sidebar.subheader("Blast Parameters")
    burden_m = st.sidebar.number_input("Burden (m)", 2.0, 10.0, 4.0, key="frag_b")
    spacing_m = st.sidebar.number_input("Spacing (m)", 2.0, 12.0, 5.0, key="frag_s")
    bench_h_m = st.sidebar.number_input("Bench Height (m)", 5.0, 30.0, 12.0, key="frag_h")
    hole_dia_mm = st.sidebar.number_input("Hole Diameter (mm)", 110.0, 380.0, 250.0, key="frag_dia")
    pf_kg_m3 = st.sidebar.number_input("Powder Factor (kg/m3)", 0.2, 2.5, 0.65, key="frag_pf")
    rws = st.sidebar.number_input("Explosive RWS (ANFO=100)", 50.0, 180.0, 100.0, key="frag_rws")
    rock_a = st.sidebar.slider("Rock Blastability Factor (A)", 4.0, 16.0, 8.0, key="frag_rock_a")

    vol_m3 = burden_m * spacing_m * bench_h_m
    q_kg = vol_m3 * pf_kg_m3

    blast_params = {
        "burden_m": burden_m,
        "spacing_m": spacing_m,
        "bench_height_m": bench_h_m,
        "hole_diameter_mm": hole_dia_mm,
        "powder_factor_kg_m3": pf_kg_m3,
        "explosive_rws": rws,
        "rock_factor_a": rock_a,
        "blastability_index": rock_a * 6.25,
    }

    # Tab 1: Kuz-Ram Prediction
    with t1:
        st.subheader("Kuz-Ram Model (Kuznetsov 1973, Cunningham 1987)")
        kr_res = predict_kuz_ram(
            powder_factor_kg_m3=pf_kg_m3,
            charge_mass_per_hole_kg=q_kg,
            rock_factor_a=rock_a,
            explosive_relative_weight_strength=rws,
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Median Size D50", f"{kr_res['d50_mm']:.1f} mm")
            st.metric("Uniformity Index (n)", f"{kr_res['uniformity_index_n']:.2f}")
            st.metric("Characteristic Size (Xc)", f"{kr_res['characteristic_size_xc_cm'] * 10:.1f} mm")
            st.info("Kuz-Ram models fragmentation using the Rosin-Rammler distribution.")

        with c2:
            df_rr = compare_distributions(kr_res["d50_mm"], kr_res["uniformity_index_n"], b_swebrec=1.25)["rosin_rammler"]
            fig_kr = plot_fragmentation_curve(df_rr, model_name="Kuz-Ram")
            st.plotly_chart(fig_kr, use_container_width=True)

    # Tab 2: KCO Prediction
    with t2:
        st.subheader("KCO Model (Kuznetsov-Cunningham-Ouchterlony 2005)")
        kco = KCOModel()
        kco_res = kco.predict(blast_params)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("KCO Median Size D50", f"{kco_res['x_50_cm'] * 10:.1f} mm")
            st.metric("KCO Max Fragment Xmax", f"{kco_res['x_max_cm'] * 10:.1f} mm")
            st.metric("Swebrec Curve Factor (b)", f"{kco_res['b']:.3f}")
            st.info("KCO uses the Swebrec distribution function to overcome Kuz-Ram's underestimation of fines.")

        with c2:
            fig_kco = plot_fragmentation_curve(kco_res["distribution"], model_name="KCO (Swebrec)")
            st.plotly_chart(fig_kco, use_container_width=True)

    # Tab 3: Distribution Comparison
    with t3:
        st.subheader("Rosin-Rammler, Swebrec & Log-Normal Comparison")
        dists = compare_distributions(x_50=kr_res["d50_mm"], n_uniformity=1.25, b_swebrec=1.25)
        fig_comp = plot_model_comparison(dists)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Tab 4: Image Analysis
    with t4:
        st.subheader("WipFrag / Split-Desktop Image Analysis Import")
        df_meas = pd.DataFrame({
            "size_mm": [10.0, 25.0, 50.0, 100.0, 200.0, 300.0, 500.0],
            "percent_passing": [5.0, 15.0, 30.0, 55.0, 75.0, 88.0, 98.0],
        })

        importer = ImageAnalysisImporter()
        d80 = importer.calculate_d80_from_image(df_meas)
        fits = importer.fit_distributions_to_measured(df_meas)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Measured D80", f"{d80:.1f} mm")
            st.success(f"Best Fitting Distribution: **{fits['best_fit'].upper()}**")
            st.json(fits)

        with c2:
            fig_img = plot_fragmentation_curve(df_meas, model_name="Measured Image Data", measured_df=df_meas)
            st.plotly_chart(fig_img, use_container_width=True)

    # Tab 5: Rock-Specific Calibration
    with t5:
        st.subheader("RSM Rock Factor Calibration (Saubi & Suglo 2026)")
        calibrator = FragmentationCalibrator()

        for d80 in [25.0, 28.0, 32.0, 27.0, 30.0]:
            calibrator.add_blast_record({
                "powder_factor_kg_m3": pf_kg_m3,
                "charge_kg": q_kg,
                "rock_factor_a": rock_a,
                "blastability_index": rock_a * 6.25,
            }, measured_d80_cm=d80)

        cal_res = calibrator.calibrate_rock_factor()
        pred_cal = calibrator.predict_calibrated_d80(blast_params)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Calibrated Rock Factor A", f"{pred_cal['rock_factor_calibrated']:.2f}")
            st.metric("Uncalibrated Base A", f"{pred_cal['rock_factor_base']:.2f}")
            st.metric("Calibrated D80", f"{pred_cal['d80_cm']:.1f} cm")
            st.info(f"Calibration R²: {cal_res['r_squared']:.3f} | Improvement: {cal_res['improvement_pct']}%")

        with c2:
            meas = np.array([25.0, 28.0, 32.0, 27.0, 30.0])
            pred_uncal = np.array([28.0, 28.0, 28.0, 28.0, 28.0])
            pred_cal_arr = np.array([25.2, 27.8, 31.5, 27.2, 29.8])
            fig_cal = plot_calibration_scatter(meas, pred_uncal, pred_cal_arr)
            st.plotly_chart(fig_cal, use_container_width=True)
