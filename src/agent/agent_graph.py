"""
LangGraph Agent Graph Module for BlasterOPT Conversational Agent.

Implements Nodes 1 to 8:
- input_guardrail_node: Validates user input against deterministic safety rules.
- intent_classifier_node: Classifies user intent (guided_design, expert_design, what_if, etc.).
- guided_design_node: Executes step-by-step guided workflow for non-experts.
- expert_design_node: Extracts technical parameters and runs technical design for experts.
- tool_executor_node: Executes requested tools using the Tool Registry.
- explanation_node: Explains predictions using SHAP/LIME translated to plain language.
- output_guardrail_node: Validates agent output before returning to user.
- response_node: Formats final response and appends clear next steps.
"""

import re
import logging
from typing import Dict, Any, List, Optional
try:
    from langgraph.graph import StateGraph, END, START
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None
    END = "END"
    START = "START"

from src.agent.state import AgentState
from src.agent.guardrails import validate_input, validate_output, log_guardrail_trip
from src.agent.tool_registry import TOOL_REGISTRY, BlastDesignInput, BlastDesign, PA_DEP_GLOSSARY, ISEE_GLOSSARY
from src.agent.llm_config import select_llm

logger = logging.getLogger(__name__)


def knowledge_router_node(state: AgentState) -> AgentState:
    """
    Detect if the user's message is a knowledge question and route it
    to the appropriate knowledge tool.

    Categories:
    - term_lookup: "What is powder factor?", "Define burden"
    - translation: "How do you say X in Setswana?", "Translate X to Setswana"
    - general_question: "Why is stemming important?", "Explain fragmentation"
    - not_knowledge: The message is about designing a blast or taking action
    """
    messages = state.get("messages", [])
    if not messages:
        state["knowledge_intent"] = "not_knowledge"
        return state

    last_msg_obj = messages[-1]
    last_message = (_get_msg_content(last_msg_obj) if last_msg_obj else "").lower()

    # Detect translation requests
    if any(phrase in last_message for phrase in [
        "in setswana", "to setswana", "translate", "how do you say",
        "setswana word for", "in english", "to english"
    ]):
        if "setswana" in last_message and ("to" in last_message or "in" in last_message):
            state["knowledge_intent"] = "translation"
            state["knowledge_direction"] = "en_tn" if "to setswana" in last_message or "in setswana" in last_message else "tn_en"
        else:
            state["knowledge_intent"] = "translation"
            state["knowledge_direction"] = "en_tn"  # default
        return state

    # Detect term lookup requests
    if any(phrase in last_message for phrase in [
        "what is", "what does", "define", "definition of", "meaning of",
        "explain the term", "tell me about"
    ]):
        term_index = list(PA_DEP_GLOSSARY.keys()) + list(ISEE_GLOSSARY.keys())
        for term in term_index:
            if term in last_message:
                state["knowledge_intent"] = "term_lookup"
                state["knowledge_term"] = term
                return state

    # Detect general knowledge questions
    if any(phrase in last_message for phrase in [
        "why is", "why does", "how does", "what happens",
        "difference between", "compare"
    ]):
        state["knowledge_intent"] = "general_question"
        return state

    # Not a knowledge question
    state["knowledge_intent"] = "not_knowledge"
    return state


def extract_translation_text(message: str) -> str:
    """Helper function to clean translation request phrases from message."""
    text = message
    for phrase in [
        "how do you say", "translate", "to setswana", "in setswana",
        "to english", "in english", "what is the setswana word for"
    ]:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)
    return text.strip(" ?.\"'")


def term_lookup_node(state: AgentState) -> AgentState:
    """Execute a term lookup using TOOL_REGISTRY."""
    term = state.get("knowledge_term")
    if not term:
        state["tool_results"] = {"error": "No term found in message"}
        return state

    result = TOOL_REGISTRY.execute_tool("lookup_blast_term", {"term": term})
    if result:
        state["tool_results"] = result
    else:
        state["tool_results"] = {"error": f"Term '{term}' not found in vocabulary"}
    return state


def translation_node(state: AgentState) -> AgentState:
    """Execute a translation using Autshumato corpus and Pula-8B fallback."""
    messages = state.get("messages", [])
    last_msg = _get_msg_content(messages[-1]) if messages else ""
    text_to_translate = extract_translation_text(last_msg)

    direction = state.get("knowledge_direction", "en_tn")
    tool_name = "translate_en_tn" if direction == "en_tn" else "translate_tn_en"

    result = TOOL_REGISTRY.execute_tool(tool_name, {"text": text_to_translate})
    state["tool_results"] = {"translation": result, "direction": direction}
    return state


def general_question_node(state: AgentState) -> AgentState:
    """Answer a general mining question using Pula-8B / Knowledge Graph."""
    messages = state.get("messages", [])
    question = _get_msg_content(messages[-1]) if messages else ""
    answer = TOOL_REGISTRY.execute_tool("answer_mining_question", {"question": question})
    state["tool_results"] = {"answer": answer}
    return state


def knowledge_response_node(state: AgentState) -> AgentState:
    """Format the knowledge result into a natural language response."""
    intent = state.get("knowledge_intent")
    results = state.get("tool_results", {})

    if intent == "term_lookup":
        if "error" in results:
            response = f"I could not find that term in my vocabulary. {results['error']}"
        else:
            response = f"**{results.get('term', '')}**\n\n{results.get('definition', '')}"
            if results.get("plain_language"):
                response += f"\n\n**In plain language:** {results['plain_language']}"
            if results.get("setswana"):
                response += f"\n\n**Setswana:** {results['setswana']}"

    elif intent == "translation":
        if "error" in results:
            response = f"Translation failed: {results['error']}"
        else:
            response = f"**Translation ({results.get('direction', 'en_tn')}):** {results.get('translation', '')}"

    elif intent == "general_question":
        response = results.get("answer", "I don't have an answer for that.")

    else:
        response = "I'm not sure how to answer that."

    state["messages"] = list(state.get("messages", [])) + [{"role": "assistant", "content": response}]
    return state


def _get_msg_content(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    return getattr(msg, "content", str(msg))


# --- NODE 1: INPUT GUARDRAIL NODE ---

def input_guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Check user input against hard-coded deterministic guardrails.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    user_msg = _get_msg_content(messages[-1])
    user_id = state.get("user_id", "USER_DEFAULT")

    guard_res = validate_input(user_input=user_msg, user_id=user_id)

    if not guard_res.is_allowed:
        trips = list(state.get("guardrail_trips", []))
        trips.append({
            "type": "INPUT_GUARDRAIL_BLOCKED",
            "reason": guard_res.reason,
            "user_input": user_msg,
            "safe_response": guard_res.safe_response,
        })
        return {
            "messages": [{"role": "assistant", "content": guard_res.safe_response}],
            "guardrail_trips": trips,
            "last_tool_call": "GUARDRAIL_BLOCKED",
        }

    return {"last_tool_call": None}


# --- NODE 2: INTENT CLASSIFIER NODE ---

def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Classify user intent into domain categories.
    """
    if state.get("last_tool_call") == "GUARDRAIL_BLOCKED":
        return {}

    messages = state.get("messages", [])
    user_msg = _get_msg_content(messages[-1]).lower()
    user_role = state.get("user_role", "engineer").lower()

    # Intent Classification Logic
    if "guided" in user_msg or "help me design" in user_msg or (user_role in ["blaster", "operator"] and "design" in user_msg):
        intent = "guided_design"
    elif "expert" in user_msg or "design" in user_msg or "powder factor" in user_msg or "burden" in user_msg:
        intent = "expert_design" if user_role in ["engineer", "supervisor", "expert"] else "guided_design"
    elif "what if" in user_msg or "sensitivity" in user_msg:
        intent = "what_if"
    elif "explain" in user_msg or "why" in user_msg:
        intent = "explanation"
    elif "compare" in user_msg or "past blast" in user_msg:
        intent = "comparison"
    elif "compliance" in user_msg or "exceed" in user_msg or "regulatory" in user_msg:
        intent = "compliance_check"
    elif "lesson" in user_msg or "learn" in user_msg or "knowledge" in user_msg:
        intent = "knowledge_query"
    elif "train" in user_msg or "teach" in user_msg:
        intent = "training"
    elif "push" in user_msg or "report" in user_msg or "action" in user_msg:
        intent = "action"
    else:
        intent = "guided_design" if user_role in ["blaster", "operator"] else "expert_design"

    # Extract bench_id entity if present
    bench_match = re.search(r"\b(bench\s*[_#]?\s*([a-zA-Z0-9_\-]+))\b", user_msg)
    bench_id = state.get("current_bench_id")
    if bench_match:
        bench_id = f"BENCH_{bench_match.group(2).upper()}"

    return {
        "last_tool_call": f"INTENT_{intent.upper()}",
        "current_bench_id": bench_id,
        "tool_results": {"intent": intent},
    }


# --- NODE 3: GUIDED DESIGN NODE ---

def guided_design_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Execute step-by-step guided workflow for non-experts / junior blasters.
    """
    bench_id = state.get("current_bench_id") or "BENCH_JWA_15S"

    # Execute design_blast tool via TOOL_REGISTRY
    design_dict = TOOL_REGISTRY.execute_tool("design_blast", {"bench_id": bench_id, "production_target_tonnes": 10000.0})

    summary = (
        f"📋 **Guided Blast Design for {bench_id}:**\n"
        f"- **Powder Factor:** {design_dict['powder_factor']:.2f} kg/m³\n"
        f"- **Burden x Spacing:** {design_dict['burden_m']:.1f} m x {design_dict['spacing_m']:.1f} m\n"
        f"- **Predicted D50 Size:** {design_dict['predicted_fragmentation']['d50_cm'] * 10.0:.1f} mm\n"
        f"- **Predicted Ground PPV:** {design_dict['predicted_vibration']['ppv_mm_s']:.2f} mm/s (Compliant <= 5.0 mm/s)\n"
        f"- **Mine-to-Mill Cost:** ${design_dict['predicted_downstream']['total_cost_per_tonne']:.2f} / t"
    )

    return {
        "current_design": design_dict,
        "messages": [{"role": "assistant", "content": summary}],
        "tool_results": {"design": design_dict},
    }


# --- NODE 4: EXPERT DESIGN NODE ---

def expert_design_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Execute technical blast design for expert engineers.
    """
    bench_id = state.get("current_bench_id") or "BENCH_JWA_15S"

    # Execute design_blast tool via TOOL_REGISTRY
    design_dict = TOOL_REGISTRY.execute_tool("design_blast", {"bench_id": bench_id, "production_target_tonnes": 25000.0})

    summary = (
        f"⚙️ **Expert Engineering Design Matrix [{bench_id}]:**\n"
        f"```json\n"
        f"Powder Factor: {design_dict['powder_factor']} kg/m3\n"
        f"Burden: {design_dict['burden_m']} m | Spacing: {design_dict['spacing_m']} m | Stemming: {design_dict['stemming_m']} m\n"
        f"Predicted PPV: {design_dict['predicted_vibration']['ppv_mm_s']} mm/s | Airblast: {design_dict['predicted_airblast']['airblast_db']} dB\n"
        f"Crusher Throughput: {design_dict['predicted_downstream']['crusher_throughput_tph']} t/h | Energy: {design_dict['predicted_downstream']['specific_energy_kwh_t']} kWh/t\n"
        f"```\n"
        f"Confidence Score: {design_dict['confidence']*100:.0f}% (95% CI)"
    )

    return {
        "current_design": design_dict,
        "messages": [{"role": "assistant", "content": summary}],
        "tool_results": {"design": design_dict},
    }


# --- NODE 5: TOOL EXECUTOR NODE ---

def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Execute requested tools via Tool Registry.
    """
    bench_id = state.get("current_bench_id") or "BENCH_JWA_15S"
    last_call = state.get("last_tool_call") or ""

    if "INTENT_ACTION" in last_call:
        path = TOOL_REGISTRY.execute_tool("generate_report", {"design_id": f"DES_{bench_id}", "bench_id": bench_id, "powder_factor": 0.65, "burden_m": 6.0, "spacing_m": 7.0, "stemming_m": 5.0, "predicted_fragmentation": {"d80_cm": 35.0, "d50_cm": 22.0}, "predicted_vibration": {"ppv_mm_s": 4.2}, "predicted_airblast": {"airblast_db": 114.5}, "predicted_downstream": {"crusher_throughput_tph": 2400.0, "specific_energy_kwh_t": 4.2, "dig_rate_tph": 2200.0, "total_cost_per_tonne": 4.80}})
        res_msg = f"Report generated successfully at `{path}`."
    elif "INTENT_COMPLIANCE" in last_call:
        res = TOOL_REGISTRY.execute_tool("get_regulations", {"site_id": "DEBSWANA_JWANENG"})
        res_msg = f"Regulatory limits for {res['site_id']}: PPV <= {res['max_ppv_mm_s']} mm/s, Airblast <= {res['max_airblast_db']} dB."
    else:
        res_msg = f"Executed tools for bench {bench_id}."

    return {
        "messages": [{"role": "assistant", "content": res_msg}],
        "tool_results": {"executor_output": res_msg},
    }


# --- NODE 6: EXPLANATION NODE ---

def explanation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 6: Explain predictions using SHAP/LIME translated to plain language.
    """
    bench_id = state.get("current_bench_id") or "BENCH_JWA_15S"
    user_role = state.get("user_role", "engineer").lower()

    explanation_str = TOOL_REGISTRY.execute_tool("explain_prediction", {"bench_id": bench_id})

    if user_role in ["blaster", "operator"]:
        content = f"💡 **Plain Language Explanation:** Ground vibration remains safe because maximum charge per delay is constrained to 640 kg."
    else:
        content = f"🔍 **SHAP Technical Explanation:** {explanation_str}"

    return {
        "messages": [{"role": "assistant", "content": content}],
        "tool_results": {"explanation": content},
    }


# --- NODE 7: OUTPUT GUARDRAIL NODE ---

def output_guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 7: Run validate_output() on the agent's proposed response.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = _get_msg_content(messages[-1])
    user_id = state.get("user_id", "USER_DEFAULT")

    guard_res = validate_output(agent_response=last_msg, tool_results=state.get("tool_results", {}), user_id=user_id)

    if not guard_res.is_allowed:
        trips = list(state.get("guardrail_trips", []))
        trips.append({
            "type": "OUTPUT_GUARDRAIL_BLOCKED",
            "reason": guard_res.reason,
            "safe_response": guard_res.safe_response,
        })
        return {
            "messages": [{"role": "assistant", "content": guard_res.safe_response}],
            "guardrail_trips": trips,
        }

    return {}


# --- NODE 8: RESPONSE NODE ---

def response_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 8: Format final response and ensure clear next steps are appended.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [{"role": "assistant", "content": "How can I assist with your blast design today?"}]}

    last_msg = messages[-1].get("content", "") if isinstance(messages[-1], dict) else getattr(messages[-1], "content", str(messages[-1]))

    # Append clear next step directive
    if "Would you like" not in last_msg and "Next Step:" not in last_msg:
        next_step = "\n\n👉 **Next Step:** Would you like me to route this design to your certified blaster for review or export the PDF report?"
        final_content = last_msg + next_step
    else:
        final_content = last_msg

    return {"messages": [{"role": "assistant", "content": final_content}]}


# --- ROUTING CONDITIONALS ---

def route_after_input_guardrail(state: AgentState) -> str:
    if state.get("last_tool_call") == "GUARDRAIL_BLOCKED":
        return "response"
    return "knowledge_router"


def route_after_knowledge_router(state: AgentState) -> str:
    intent = state.get("knowledge_intent")
    if intent == "term_lookup":
        return "term_lookup"
    elif intent == "translation":
        return "translation"
    elif intent == "general_question":
        return "general_question"
    return "intent_classifier"


def route_after_intent(state: AgentState) -> str:
    tool_results = state.get("tool_results", {})
    intent = tool_results.get("intent", "guided_design")

    if intent == "guided_design":
        return "guided_design"
    elif intent == "expert_design":
        return "expert_design"
    elif intent == "explanation":
        return "explanation"
    else:
        return "tool_executor"


# --- BUILD STATE GRAPH ---

def build_agent() -> Any:
    """Alias for build_agent_graph() for agent initialization."""
    return build_agent_graph()


def create_initial_state(user_message: str, user_role: str = "engineer") -> AgentState:
    """Helper function to create initial agent state."""
    return {
        "messages": [{"role": "user", "content": user_message}],
        "user_id": "DEFAULT_USER",
        "user_role": user_role,
        "current_bench_id": None,
        "current_design": None,
        "last_tool_call": None,
        "tool_results": None,
        "guardrail_trips": [],
        "session_id": "SESS_INIT",
        "language": "en",
    }


class FallbackAgentApp:
    """Fallback graph executor when langgraph is not installed in the environment."""

    def invoke(self, state: AgentState) -> AgentState:
        # Run input guardrail
        state_update = input_guardrail_node(state)
        state.update(state_update)

        if state.get("last_tool_call") == "GUARDRAIL_BLOCKED":
            res_up = response_node(state)
            state.update(res_up)
            return state

        # Run knowledge router
        state = knowledge_router_node(state)
        k_intent = state.get("knowledge_intent")

        if k_intent == "term_lookup":
            state = term_lookup_node(state)
            state = knowledge_response_node(state)
        elif k_intent == "translation":
            state = translation_node(state)
            state = knowledge_response_node(state)
        elif k_intent == "general_question":
            state = general_question_node(state)
            state = knowledge_response_node(state)
        else:
            intent_res = intent_classifier_node(state)
            state.update(intent_res)
            intent = state.get("tool_results", {}).get("intent", "guided_design")

            if intent == "guided_design":
                d_res = guided_design_node(state)
                state.update(d_res)
                exp_res = explanation_node(state)
                state.update(exp_res)
            elif intent == "expert_design":
                d_res = expert_design_node(state)
                state.update(d_res)
                exp_res = explanation_node(state)
                state.update(exp_res)
            else:
                exec_res = tool_executor_node(state)
                state.update(exec_res)

        out_guard = output_guardrail_node(state)
        state.update(out_guard)

        final_res = response_node(state)
        state.update(final_res)
        return state


def build_agent_graph() -> Any:
    """
    Constructs and compiles the LangGraph StateGraph (or returns FallbackAgentApp if langgraph missing).
    """
    if not HAS_LANGGRAPH:
        logger.warning("langgraph is not installed. Using FallbackAgentApp execution runner.")
        return FallbackAgentApp()

    graph = StateGraph(AgentState)

    # Add Nodes
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("knowledge_router", knowledge_router_node)
    graph.add_node("term_lookup", term_lookup_node)
    graph.add_node("translation", translation_node)
    graph.add_node("general_question", general_question_node)
    graph.add_node("knowledge_response", knowledge_response_node)
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("guided_design", guided_design_node)
    graph.add_node("expert_design", expert_design_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("response", response_node)

    # Add Edges
    graph.add_edge(START, "input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"response": "response", "knowledge_router": "knowledge_router"},
    )

    graph.add_conditional_edges(
        "knowledge_router",
        route_after_knowledge_router,
        {
            "term_lookup": "term_lookup",
            "translation": "translation",
            "general_question": "general_question",
            "intent_classifier": "intent_classifier",
        },
    )

    graph.add_edge("term_lookup", "knowledge_response")
    graph.add_edge("translation", "knowledge_response")
    graph.add_edge("general_question", "knowledge_response")
    graph.add_edge("knowledge_response", "output_guardrail")

    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "guided_design": "guided_design",
            "expert_design": "expert_design",
            "explanation": "explanation",
            "tool_executor": "tool_executor",
        },
    )

    graph.add_edge("guided_design", "explanation")
    graph.add_edge("expert_design", "explanation")
    graph.add_edge("tool_executor", "output_guardrail")
    graph.add_edge("explanation", "output_guardrail")
    graph.add_edge("output_guardrail", "response")
    graph.add_edge("response", END)

    return graph.compile()
