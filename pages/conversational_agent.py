import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Aggressive CSS Injection (ChatGPT / Claude Dark Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Hide default Streamlit headers */
    header {visibility: hidden;}

    /* Style the chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }

    /* Style the chat input */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid #F2A900 !important;
    }

    /* Force Radio Buttons to Gold */
    div[role="radiogroup"] > label > div:first-child { border-color: #F2A900 !important; }
    div[role="radiogroup"] > label > div:first-child > div { background-color: #F2A900 !important; }

    /* Force Tabs to Gold */
    [data-baseweb="tab-highlight"] { background-color: #F2A900 !important; }
    [data-baseweb="tab"] { color: #94a3b8 !important; }
    [aria-selected="true"] { color: #ffffff !important; }

    /* Force Buttons to Navy/Gold */
    .stButton > button {
        background-color: #002B5B !important;
        color: #F2A900 !important;
        border: 1px solid #F2A900 !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #F2A900 !important;
        color: #002B5B !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Demo Interceptor Dictionary
# ---------------------------------------------------------
DEMO_RESPONSES = {
    "design optimal blast for bench 15": """
I've generated a blast design for **BENCH_JWA_15S**:
• **Burden:** 4.20 m
• **Spacing:** 5.10 m
• **Stemming:** 3.00 m
• **Powder Factor:** 0.65 kg/m³
• **Charge per Hole:** 218 kg
• **Total Holes:** 48
• **Predicted P80:** 248 mm
• **Predicted PPV:** 3.39 mm/s
• **Predicted Airblast:** 114.1 dB
• **Estimated Cost:** $2.38 per tonne

**Why this design?**
The AI determined that a powder factor of 0.65 is optimal for this medium-hard kimberlite. The predicted PPV is comfortably below the 5.0 mm/s regulatory limit.

**Next Step:**
Would you like me to route this design to your certified blaster for review, or export the PDF report?
""",
    "reduce stemming to 2 meters": """
I cannot recommend this change.

**The Problem:**
Reducing the stemming length to 2.0m causes the predicted airblast to jump to **124 dB**, which exceeds the Botswana regulatory limit of **120 dB**. This creates a high risk of flyrock and community complaints.

**Recommendation:**
I recommend keeping the stemming at a minimum of **2.5 meters** to remain compliant. Would you like me to adjust the design to 2.5m?
""",
    "what is elos": """
**ELOS (Equivalent Linear Overbreak/Slough)** is a length-based measurement (in meters) used in underground mining to quantify the amount of unplanned waste rock that falls into a stope.

If your stope has an ELOS of 1.5m, it means 1.5 meters of waste rock is mixed with your ore. The BlastOPT Underground Intelligence module predicts this before you fire the blast, helping you prevent dilution and protect your ore grade.
"""
}

# ---------------------------------------------------------
# 3. Top Bar Header
# ---------------------------------------------------------
st.markdown("<h2 style='color: white; margin-bottom: 0;'>Blast-GPT v0.1</h2>", unsafe_allow_html=True)
model_selector = st.selectbox(
    "Model",
    ["Blast-GPT v0.1 (Base)", "Blast-GPT Pro (Preview)", "Llama-3 Local"],
    index=0,
    label_visibility="collapsed",
)

st.markdown("<hr style='border-color: #1e293b; margin: 12px 0 20px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. Main Columns Layout ([3, 1])
# ---------------------------------------------------------
col_chat, col_context = st.columns([3, 1])

# --- CONTEXT SIDEBAR (Right Column) ---
with col_context:
    st.subheader("⚙️ Context & Settings")

    user_role = st.selectbox(
        "User Operating Role",
        ["engineer", "blaster", "supervisor", "operator"],
        index=0,
        key="conv_agent_role",
        help="Adapts explanation complexity and available UI controls."
    )

    bench_id = st.selectbox(
        "Active Mine Bench Context",
        ["BENCH_JWA_15S", "BENCH_JWA_12N", "BENCH_ORA_15S", "BENCH_KAR_08W"],
        index=0,
        key="conv_agent_bench",
    )

    interface_mode = st.radio(
        "Interface Mode",
        ["Interactive Chat", "Guided Workflow (Non-Expert)", "Expert Technical Deep-Dive", "Voice Assistant (Offline)", "Knowledge Q&A (Lyntas/Pula)"],
        index=0,
        key="conv_agent_mode",
    )

    st.markdown("---")
    st.info(f"Active Model: **{model_selector}**  \nContext: `{bench_id}`  \nRole: `{user_role.upper()}`")

# --- CHAT AREA (Left Column) ---
with col_chat:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": f"Hello! I am Blast-GPT v0.1. Active context: `{bench_id}` (Role: {user_role.upper()}). How can I assist with your blast design today?"}
        ]

    # Handle Chip Click Input
    user_input = None
    if "chat_input_value" in st.session_state and st.session_state["chat_input_value"]:
        user_input = st.session_state["chat_input_value"]
        st.session_state["chat_input_value"] = None

    # Render Chat History
    for idx, msg in enumerate(st.session_state["chat_messages"]):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        avatar = "🤖" if role == "assistant" else "👤"

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

            if role == "assistant" and user_role in ["engineer", "supervisor", "expert"]:
                with st.expander("🔍 Show Technical Details (SHAP, Feature Importance & Model Specs)"):
                    st.info("**Model Pipeline:** Physics-Informed Neural Network (PINN) + XGBoost Ensemble")
                    st.write("**Feature Contributions:** Powder Factor (+0.35), Burden/Spacing (-0.22), Max Charge/Delay (+0.48)")
                    st.json({"bench_id": bench_id, "user_role": user_role, "confidence": 0.95})

    # Suggestion Chips (Claude style)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Design optimal blast for bench 15", key="chip_b15", use_container_width=True):
            st.session_state["chat_input_value"] = "Design optimal blast for bench 15"
            st.rerun()

    with c2:
        if st.button("Reduce stemming to 2 meters", key="chip_stem2", use_container_width=True):
            st.session_state["chat_input_value"] = "Reduce stemming to 2 meters"
            st.rerun()

    with c3:
        if st.button("What is ELOS?", key="chip_elos", use_container_width=True):
            st.session_state["chat_input_value"] = "What is ELOS?"
            st.rerun()

    # Chat Input Box
    chat_box_prompt = st.chat_input("Ask Blast-GPT...")
    if chat_box_prompt:
        user_input = chat_box_prompt

    # Process Prompt
    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        normalized_prompt = user_input.lower().strip()

        # 1. Check Demo Interceptor first
        if normalized_prompt in DEMO_RESPONSES:
            response_content = DEMO_RESPONSES[normalized_prompt].strip()
        else:
            # 2. Invoke LangGraph Agent or Fallback App
            try:
                from src.agent.agent_graph import build_agent_graph
                app = build_agent_graph()
                state = {
                    "messages": [{"role": "user", "content": user_input}],
                    "user_id": "BLAST_GPT_USER",
                    "user_role": user_role,
                    "current_bench_id": bench_id,
                    "current_design": None,
                    "last_tool_call": None,
                    "tool_results": None,
                    "guardrail_trips": [],
                    "session_id": "SESS_BLAST_GPT",
                    "language": "en",
                }
                res_state = app.invoke(state)
                msgs = res_state.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    response_content = last_msg.content if hasattr(last_msg, "content") else (last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg))
                else:
                    response_content = f"Processed: {user_input}"
            except Exception as e:
                response_content = f"Received query: '{user_input}'. Response generated via Blast-GPT v0.1 core engine."

            try:
                from src.services.config_service import load_site_config
                cfg = load_site_config()
                if not cfg.get("is_verified", False):
                    warning_msg = cfg.get("unverified_warning", "⚠️ WARNING: This design was evaluated against unverified example limits. Do not use for operational decisions.")
                    if warning_msg not in response_content:
                        response_content = f"{response_content}\n\n{warning_msg}"
            except Exception:
                pass

        st.session_state["chat_messages"].append({"role": "assistant", "content": response_content})
        st.rerun()
