import streamlit as st
from datetime import datetime
import pandas as pd
from src.synthetic_data import generate_synthetic_blast_data, validate_synthetic_data
from src.data_ingestion import load_real_blast_data, clean_and_preprocess, engineer_features, dataframe_fingerprint

st.title("Data Ingestion & Generator")

tab1, tab2 = st.tabs(["⚡ Generate Synthetic Blast Logs", "📁 Upload Custom Blast CSV"])

with tab1:
    st.subheader("Physics-Guided Synthetic Blast Data Generator")
    col_gen1, col_gen2 = st.columns(2)

    with col_gen1:
        num_samples = st.slider("Number of Blast Logs", 50, 2000, 400, step=50)
        seed_val = st.number_input("Random Seed", value=42)

    with col_gen2:
        rock_factor_min, rock_factor_max = st.slider(
            "Rock Blastability Factor (A) Range", 4.0, 16.0, (6.0, 12.0)
        )

    if st.button("Generate Synthetic Dataset", type="primary"):
        new_df = generate_synthetic_blast_data(
            num_samples=num_samples,
            seed=int(seed_val),
            rock_factor_range=(rock_factor_min, rock_factor_max),
        )
        processed_df = engineer_features(clean_and_preprocess(new_df))
        validate_synthetic_data(processed_df)

        keys_to_clear = [
            "df", "dataset", "df_clean", "synthetic_df", "uploaded_df",
            "regenerated_df", "data_ready", "data_fingerprint",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["trained_models"] = {}
        st.session_state["trained_model_metadata"] = {}
        st.info("Old trained models cleared. Retrain on the new dataset.")

        st.session_state["df"] = processed_df
        st.session_state["data_ready"] = True
        st.session_state["data_source"] = "synthetic"
        st.session_state["data_rows"] = processed_df.shape[0]
        st.session_state["data_cols"] = processed_df.shape[1]
        st.session_state["data_fingerprint"] = dataframe_fingerprint(processed_df)
        st.session_state["data_generated_at"] = datetime.now().isoformat()
        st.session_state["data_loaded_at"] = datetime.now().isoformat()
        st.session_state["data_details"] = f"Synthetic {processed_df.shape[0]} rows, seed={seed_val}"

        st.success(
            f"✅ Generated {processed_df.shape[0]} rows, {processed_df.shape[1]} columns. "
            f"Fingerprint: `{st.session_state['data_fingerprint']}`"
        )
        st.rerun()

    if st.session_state.get("data_ready", False) and st.session_state.get("data_source") == "synthetic":
        curr_synth_df = st.session_state["df"]
        st.divider()
        st.success(
            f"✅ Active Synthetic Dataset: {curr_synth_df.shape[0]} rows × {curr_synth_df.shape[1]} columns  \n"
            f"Fingerprint: `{st.session_state.get('data_fingerprint')}` | "
            f"Generated at: {st.session_state.get('data_generated_at')}"
        )
        st.caption("First 15 records preview:")
        st.dataframe(curr_synth_df.head(15), use_container_width=True)

with tab2:
    st.subheader("Upload Real Mine Production Data vs Synthetic Data")
    uploaded_file = st.file_uploader("Upload Real Mine CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            temp_path = "data/raw/uploaded_real_blast_data.csv"
            import os
            os.makedirs("data/raw", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("Uploaded CSV Raw Preview:")
            st.dataframe(pd.read_csv(temp_path).head(5), use_container_width=True)

            if st.button("Process & Load Real Mine Data", type="primary"):
                real_processed_df = load_real_blast_data(
                    filepath=temp_path,
                    anomaly_log_path="data/processed/data_anomalies.log"
                )
                validate_synthetic_data(real_processed_df)

                keys_to_clear = [
                    "df", "dataset", "df_clean", "synthetic_df", "uploaded_df",
                    "regenerated_df", "data_ready", "data_fingerprint",
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                st.session_state["trained_models"] = {}
                st.session_state["trained_model_metadata"] = {}
                st.info("Old trained models cleared. Retrain on the new dataset.")

                st.session_state["df"] = real_processed_df
                st.session_state["data_ready"] = True
                st.session_state["data_source"] = "uploaded"
                st.session_state["data_rows"] = real_processed_df.shape[0]
                st.session_state["data_cols"] = real_processed_df.shape[1]
                st.session_state["data_fingerprint"] = dataframe_fingerprint(real_processed_df)
                st.session_state["data_generated_at"] = datetime.now().isoformat()
                st.session_state["data_loaded_at"] = datetime.now().isoformat()
                st.session_state["data_details"] = f"Uploaded CSV {real_processed_df.shape[0]} rows"

                st.success(
                    f"✅ Loaded {real_processed_df.shape[0]} rows from {uploaded_file.name}. "
                    f"Fingerprint: `{st.session_state['data_fingerprint']}`"
                )
                st.rerun()
        except Exception as e:
            st.error(f"Error processing real mine dataset: {e}")

    if st.session_state.get("data_ready", False) and st.session_state.get("data_source") == "uploaded":
        curr_up_df = st.session_state["df"]
        st.divider()
        st.success(
            f"✅ Active Uploaded Mine Dataset: {curr_up_df.shape[0]} rows × {curr_up_df.shape[1]} columns  \n"
            f"Fingerprint: `{st.session_state.get('data_fingerprint')}` | "
            f"Loaded at: {st.session_state.get('data_loaded_at')}"
        )
        st.caption("First 15 records preview:")
        st.dataframe(curr_up_df.head(15), use_container_width=True)
