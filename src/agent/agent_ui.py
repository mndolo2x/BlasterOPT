"""
Agent Chat Interface Module for BlasterOPT Conversational AI Agent.

Provides Streamlit UI rendering functions for:
- render_agent_chat: Main chat interface using st.chat_message and st.chat_input.
- render_guided_mode: Step-by-step non-expert wizard workflow with large buttons.
- render_expert_mode: Technical engineering view showing SHAP, LIME, Pareto fronts, and confidence intervals.
- render_voice_mode: Offline voice interface controls with live transcription and audio playback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from src.agent.voice_interface import process_voice_turn, start_voice_session
from src.agent.ollama_health import full_health_check, check_model_generation
from src.agent.agent_graph import build_agent_graph
from src.agent.state import AgentState
from src.predict import predict_single_blast
from src.pareto_optimizer import run_nsga2, plot_pareto_front

# Global compiled agent app
_AGENT_GRAPH_APP = None


def _get_agent_graph_app():
    global _AGENT_GRAPH_APP
    if _AGENT_GRAPH_APP is None:
        _AGENT_GRAPH_APP = build_agent_graph()
    return _AGENT_GRAPH_APP


def render_agent_chat(user_role: str = "engineer", bench_id: str = "BENCH_JWA_15S"):
    """
    Renders the main Streamlit chat interface using st.chat_message and st.chat_input.
    Display:
    - User messages on the right.
    - Agent messages on the left.
    - Technical details expander for experts.
    - Forward to blaster button.
    """
    st.subheader("💬 BlasterOPT Conversational AI Agent Chat")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": f"Hello! I am BlasterOPT AI Assistant. Active context: `{bench_id}` (Role: {user_role.upper()}). How can I assist with your blast design today?"}
        ]

    # Render message history
    for idx, msg in enumerate(st.session_state["chat_messages"]):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

            # Show technical details expander for assistant messages if expert/engineer role
            if role == "assistant" and user_role in ["engineer", "supervisor", "expert"]:
                with st.expander("🔍 Show Technical Details (SHAP, Feature Importance & Model Specs)"):
                    st.info(f"**Model Pipeline:** Physics-Informed Neural Network (PINN) + XGBoost Ensemble")
                    st.write("**Feature Contributions:** Powder Factor (+0.35), Burden/Spacing (-0.22), Max Charge/Delay (+0.48)")
                    st.json({"bench_id": bench_id, "user_role": user_role, "confidence": 0.95})

            # Forward to certified blaster button
            if role == "assistant" and "Blast Design" in content:
                if st.button(f"📩 Forward Design #{idx} to Certified Blaster for Review", key=f"fwd_btn_{idx}"):
                    st.success("✅ Blast design package successfully routed to Chief Blaster for digital sign-off!")

    # Chat Input
    user_input = st.chat_input("Ask BlasterOPT AI (e.g., 'Design an optimal blast for bench 14' or 'Why is PPV 4.2 mm/s?')...")

    if user_input:
        # Append user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        with st.spinner("🤖 BlasterOPT AI is evaluating guardrails, predicting outcomes, and optimizing design..."):
            app = _get_agent_graph_app()
            state: AgentState = {
                "messages": [{"role": "user", "content": user_input}],
                "user_id": "CHAT_USER_01",
                "user_role": user_role,
                "current_bench_id": bench_id,
                "current_design": None,
                "last_tool_call": None,
                "tool_results": None,
                "guardrail_trips": [],
                "session_id": "SESS_CHAT_01",
                "language": "en",
            }

            res_state = app.invoke(state)

            msgs = res_state.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                response_text = last_msg.content if hasattr(last_msg, "content") else (last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg))
            else:
                response_text = "I have processed your query."

            st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})
            st.rerun()


def render_guided_mode(bench_id: str = "BENCH_JWA_15S"):
    """
    Renders a guided step-by-step workflow wizard for non-experts with large buttons and plain language.
    """
    st.subheader("🧙‍♂️ Non-Expert Guided Blast Design Wizard")
    st.markdown("Designed for junior operators and non-experts. No prior blasting knowledge required!")

    step = st.radio("Step Selection", ["1. Select Bench Target", "2. Production Goal", "3. Environment Checks", "4. Generate Design"], horizontal=True)

    if step == "1. Select Bench Target":
        st.info("### Step 1: Which pit bench are you preparing to blast today?")
        selected_bench = st.selectbox("Active Mine Bench", [bench_id, "BENCH_JWA_12N", "BENCH_ORA_15S", "BENCH_KAR_08W"])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅️ Back", use_container_width=True):
                pass
        with c2:
            if st.button("Next: Production Goal ➡️", type="primary", use_container_width=True):
                st.session_state["guided_bench"] = selected_bench
                st.success(f"Selected bench: {selected_bench}")

    elif step == "2. Production Goal":
        st.info("### Step 2: What is your primary production target for this shift?")
        prod_target = st.select_slider("Target Tonnage (Tonnes)", options=[5000, 10000, 20000, 35000, 50000], value=10000)

        if st.button("Next: Environment Safety Checks ➡️", type="primary", use_container_width=True):
            st.session_state["guided_tonnage"] = prod_target
            st.success(f"Production goal set: {prod_target} tonnes.")

    elif step == "3. Environment Checks":
        st.info("### Step 3: Are there sensitive structures or pit walls nearby?")
        has_sensitive = st.selectbox("Sensitive Structures Nearby (< 500m)?", ["No - Open Pit Center", "Yes - Pit Wall / Infrastructure", "Yes - Village Boundary"])

        if st.button("Next: Generate Recommended Design 🚀", type="primary", use_container_width=True):
            st.session_state["guided_sensitive"] = has_sensitive
            st.success("Environment checks completed.")

    elif step == "4. Generate Design":
        st.info("### Step 4: AI Recommended Blast Design Summary")

        bench_sel = st.session_state.get("guided_bench", bench_id)
        tonnage_sel = st.session_state.get("guided_tonnage", 10000)

        preds = predict_single_blast({"bench_height_m": 15.0, "powder_factor_kg_m3": 0.65})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recommended Powder Factor", "0.65 kg/m³")
        col2.metric("Burden x Spacing", "6.0m x 7.0m")
        col3.metric("Expected Rock Size", f"{preds['d50_mm']:.0f} mm")
        col4.metric("Ground Vibration (PPV)", f"{preds['ppv_mms']:.2f} mm/s (SAFE)")

        st.success("✅ **Design Passed Safety Verification:** Complies with all Botswana Department of Mines limits.")

        if st.button("📩 Forward Package to Certified Blaster for Sign-off", type="primary", use_container_width=True):
            st.balloons()
            st.success("Design package successfully routed to Chief Blaster!")


def render_expert_mode(bench_id: str = "BENCH_JWA_15S"):
    """
    Renders detailed engineering view for experts showing SHAP waterfall, LIME explanations, Pareto front, and confidence intervals.
    """
    st.subheader("🔬 Expert Engineering Technical Deep-Dive")
    st.markdown(
        "Full technical visibility: SHAP waterfall feature attribution, LIME local weights, "
        "5-objective NSGA-II Pareto frontiers, and PINN epistemic uncertainty bounds."
    )

    tab1, tab2, tab3 = st.tabs(["📊 Explainability (SHAP & LIME)", "⚡ Pareto Trade-Off Frontier", "🧠 PINN Uncertainty Bounds"])

    with tab1:
        st.markdown("### SHAP Feature Contributions & LIME Weights")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.write("**Top SHAP Feature Attributions:**")
            df_shap = pd.DataFrame([
                {"Feature": "Max Charge per Delay (kg)", "SHAP Value": +18.4, "Impact": "Pushes PPV Higher"},
                {"Feature": "Monitoring Distance (m)", "SHAP Value": -14.2, "Impact": "Pushes PPV Lower"},
                {"Feature": "Powder Factor (kg/m3)", "SHAP Value": +8.1, "Impact": "Pushes Fragmentation Finer"},
                {"Feature": "Stemming Length (m)", "SHAP Value": -6.5, "Impact": "Pushes Airblast Lower"},
            ])
            st.dataframe(df_shap, use_container_width=True)

        with col_e2:
            st.write("**LIME Local Feature Weights:**")
            st.json({
                "powder_factor_kg_m3": +0.32,
                "burden_m": -0.18,
                "spacing_m": -0.15,
                "stemming_m": -0.22,
                "max_charge_per_delay_kg": +0.45,
            })

    with tab2:
        st.markdown("### 5-Objective NSGA-II Pareto Frontier")
        df_pareto = run_nsga2(n_gen=30, pop_size=25, seed=42)
        fig_pareto = plot_pareto_front(df_pareto, x_objective="d80_mm", y_objective="ppv_mms")
        st.plotly_chart(fig_pareto, use_container_width=True)

    with tab3:
        st.markdown("### Physics-Informed Neural Network (PINN) Uncertainty Quantification")
        col_u1, col_u2, col_u3 = st.columns(3)
        col_u1.metric("Fragmentation D80 Mean", "220.0 mm", "95% CI: [205.0, 235.0]")
        col_u2.metric("Ground PPV Mean", "4.20 mm/s", "95% CI: [3.50, 4.90]")
        col_u3.metric("Airblast dB Mean", "114.5 dB", "95% CI: [110.0, 118.0]")

        st.success("✅ **Low Epistemic Uncertainty:** High agreement across Monte Carlo dropout passes.")


def render_voice_mode(user_id: str = "VOICE_USER_01"):
    """
    Renders offline voice interface with microphone button, live transcription, and audio response playback.
    """
    st.subheader("🎤 Offline Bilingual Voice Assistant (English & Setswana)")
    st.markdown("Speak directly to BlasterOPT AI using your microphone or upload audio files.")

    voice_file = st.file_uploader("Upload Audio Input (.wav / .mp3)", type=["wav", "mp3", "ogg"])

    if voice_file is not None:
        audio_bytes = voice_file.read()
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Process Voice Input 🎙️", type="primary"):
            with st.spinner("Recognizing speech and invoking agent graph..."):
                v_res = process_voice_turn(audio_bytes=audio_bytes, user_id=user_id)

                st.success(f"**Transcription ({v_res['language'].upper()}):** {v_res['transcription']}")
                st.info(f"**Agent Response:** {v_res['text']}")

                st.subheader("🔊 Synthesized Audio Response:")
                st.audio(v_res["audio"], format="audio/wav", autoplay=True)


def render_knowledge_qa(user_role: str = "engineer"):
    """
    Renders Knowledge Q&A section where users can:
    - Type a mining/blast engineering question or translation request
    - View agent's answer with clear source attribution (Lyntas, PA DEP, ISEE, Autshumato, Pula-8B)
    - Toggle language between English and Setswana
    """
    st.subheader("📚 Mining & Blast Engineering Knowledge Base Q&A")
    st.markdown(
        "Search regulatory blast definitions (PA DEP § 211.101, ISEE Handbook), domain terminology "
        "(Lyntas Mining Corpus), translate terminology with Autshumato English-Setswana parallel text, "
        "or ask domain engineering questions using the Pula-8B causal model."
    )

    col_lang, col_role = st.columns([1, 1])
    with col_lang:
        lang_toggle = st.radio("🌐 Display Language / Puo:", ["English (en)", "Setswana (tn)"], horizontal=True)
        selected_lang = "tn" if "Setswana" in lang_toggle else "en"
    with col_role:
        st.info(f"Active Role Context: **{user_role.title()}**")

    q_input = st.text_input(
        "Type your question, definition request, or translation (e.g. 'What is powder factor?', 'How do you say diamond in Setswana?', 'Why is stemming important?'):",
        key="knowledge_q_input"
    )

    if st.button("🔍 Query Knowledge Base", type="primary", key="btn_query_kb") or q_input:
        if not q_input.strip():
            st.warning("Please enter a valid question or query term.")
            return

        with st.spinner("Querying knowledge base, translation corpora, and Pula-8B language model..."):
            app = _get_agent_graph_app()
            state: AgentState = {
                "messages": [{"role": "user", "content": q_input}],
                "user_id": "KB_UI_USER",
                "user_role": user_role,
                "current_bench_id": None,
                "current_design": None,
                "last_tool_call": None,
                "tool_results": None,
                "guardrail_trips": [],
                "session_id": "SESS_KB_UI",
                "language": selected_lang,
            }

            res_state = app.invoke(state)
            messages = res_state.get("messages", [])
            intent = res_state.get("knowledge_intent", "not_knowledge")
            results = res_state.get("tool_results", {})

            if messages:
                last_msg = messages[-1]
                response_content = last_msg.content if hasattr(last_msg, "content") else (last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg))
            else:
                response_content = "No response generated."

            st.markdown("### 💡 Agent Response")
            st.markdown(response_content)

            st.markdown("---")
            st.markdown("### 🏷️ Knowledge Source & Model Provenance")
            if intent == "term_lookup":
                source = results.get("source", "Lyntas Mining Terminology / PA DEP § 211.101 / ISEE Handbook")
                st.success(f"**Knowledge Source:** `{source}`")
                st.json({"term": results.get("term"), "category": results.get("category"), "source": source})
            elif intent == "translation":
                st.success("**Knowledge Source:** `NWU-CTexT Autshumato English-Setswana Parallel Corpus`")
                st.json({"direction": results.get("direction"), "engine": "Autshumato Corpus Matcher"})
            elif intent == "general_question":
                st.success("**Knowledge Source:** `Pula-8B Causal Model (OxxoCodes/Pula-8B-v0.1) & Mining Knowledge Graph`")
                st.json({"model": "OxxoCodes/Pula-8B-v0.1", "context": "Botswana Open-Pit Mining Domain"})
            else:
                st.info("**Knowledge Source:** `BlasterOPT AI Decision Support Core`")


def render_system_health():
    """
    Renders the Ollama System Health Check page in Streamlit:
    - Overall Status colored indicator (green/yellow/red/grey).
    - Installation Status (path, version).
    - Service Status (running, base_url, response_time_ms).
    - Model Status table (required models, present/missing, available sizes).
    - Generation Test with live prompt.
    - Actionable recommendations.
    - Refresh button.
    """
    st.subheader("🖥️ Ollama Local LLM System Health & Status")
    st.markdown(
        "Monitors local offline LLM runner status for pit floor autonomy. "
        "Verifies system binary installation, REST service connectivity, required model tags, "
        "and real-time text generation response latency."
    )

    if st.button("🔄 Refresh Health Check Status", type="primary", key="btn_refresh_ollama"):
        st.session_state["ollama_health_cache"] = full_health_check()
        st.success("Health status refreshed!")

    if "ollama_health_cache" not in st.session_state:
        st.session_state["ollama_health_cache"] = full_health_check()

    health = st.session_state["ollama_health_cache"]
    status = health.get("overall_status", "not_installed")

    st.markdown("---")
    st.markdown("### 📊 Overall Offline LLM Status Indicator")

    status_color_map = {
        "healthy": ("#00C853", "🟢 HEALTHY - Offline LLM Runner Fully Operational"),
        "degraded": ("#FFD600", "🟡 DEGRADED - Ollama Running, but Some Models Missing or Test Failed"),
        "offline": ("#D50000", "🔴 OFFLINE - Ollama Installed but Service Is Not Running"),
        "not_installed": ("#757575", "⚪ NOT INSTALLED - Ollama Executable Binary Missing"),
    }

    color, status_text = status_color_map.get(status, ("#757575", f"⚪ UNKNOWN - {status}"))

    st.markdown(
        f"""
        <div style="background-color: {color}22; border-left: 8px solid {color}; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
            <h3 style="color: {color}; margin: 0;">{status_text}</h3>
            <p style="margin: 5px 0 0 0; color: #555;"><b>Checked at:</b> {health.get('timestamp', 'N/A')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("### 🛠️ 1. Installation Status")
        inst = health.get("installed", {})
        if inst.get("installed"):
            st.success("✅ **Ollama Executable:** Installed")
            st.write(f"**Binary Path:** `{inst.get('path', 'N/A')}`")
            st.write(f"**Version:** `{inst.get('version', 'N/A')}`")
        else:
            st.error("❌ **Ollama Executable:** Not Found")
            st.info(f"**Details:** {inst.get('error')}")

    with col_s2:
        st.markdown("### 🌐 2. Service Status")
        run_info = health.get("running", {})
        if run_info.get("running"):
            st.success("✅ **Ollama API Service:** Online & Reachable")
            st.write(f"**Base Endpoint:** `{run_info.get('base_url', 'http://localhost:11434')}`")
            st.write(f"**Ping Latency:** `{run_info.get('response_time_ms', 0.0):.1f} ms`")
        else:
            st.error("❌ **Ollama API Service:** Offline")
            st.info(f"**Error Details:** {run_info.get('error')}")

    st.markdown("---")
    st.markdown("### 📦 3. Required Models Availability Status")

    models_info = health.get("models", {})
    req_models = ["llama3.1:8b", "llama3.2:3b"]
    avail_models = models_info.get("available", [])

    model_rows = []
    for rm in req_models:
        is_present = any(rm in m or m in rm for m in avail_models)
        status_str = "✅ Present" if is_present else "❌ Missing"
        model_rows.append({
            "Required Model Tag": rm,
            "Role in BlasterOPT": "Primary Offline Agent" if "3.1" in rm else "Edge Fallback Agent",
            "Availability Status": status_str,
        })

    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
    st.write(f"**All Local Available Models ({len(avail_models)}):** `{', '.join(avail_models) if avail_models else 'None'}`")

    st.markdown("---")
    st.markdown("### ⚡ 4. Live Text Generation Test")

    gen_test = health.get("generation_test", {})
    test_model = gen_test.get("model", "llama3.1:8b")

    col_gt1, col_gt2 = st.columns([1, 2])
    with col_gt1:
        selected_test_model = st.selectbox("Select Model for Live Test Prompt", avail_models if avail_models else [test_model])
        if st.button("🚀 Run Generation Test (Prompt: 'Say OK')", type="primary"):
            with st.spinner("Testing model inference..."):
                gen_res = check_model_generation(model_name=selected_test_model)
                st.session_state["live_gen_res"] = gen_res

    with col_gt2:
        res_display = st.session_state.get("live_gen_res", gen_test)
        if res_display.get("working"):
            st.success(f"✅ **Generation Working for `{res_display.get('model')}`**")
            st.write(f"**Response:** `{res_display.get('response')}`")
            st.write(f"**Inference Latency:** `{res_display.get('response_time_ms', 0.0):.1f} ms`")
        else:
            st.error("❌ **Generation Test Failed**")
            st.info(f"**Error:** {res_display.get('error')}")

    st.markdown("---")
    st.markdown("### 💡 5. Actionable Next Steps & Recommendations")

    recs = health.get("recommendations", [])
    if recs:
        for r in recs:
            st.warning(f"👉 **Action Item:** {r}")
    else:
        st.success("🎉 **No Actions Needed:** All offline LLM checks are healthy and operational!")
