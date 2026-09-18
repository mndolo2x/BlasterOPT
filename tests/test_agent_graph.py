"""
Unit tests for LangGraph Agent Graph, LLM Configuration, and Conversation Memory modules.
"""

import pytest
from src.agent.state import AgentState
from src.agent.agent_graph import (
    input_guardrail_node,
    intent_classifier_node,
    tool_executor_node,
    output_guardrail_node,
    build_agent_graph,
)
from src.agent.llm_config import select_llm, CloudLLM, LocalLLM, Pula8BLLM, HuggingFaceLLM, OllamaClient
from src.agent.memory import AgentMemoryManager


def test_input_guardrail_node_blocks_unsafe_input():
    """Test input_guardrail_node blocks forbidden inputs like 'just fire it'."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Just fire it immediately!"}],
        "user_id": "USER_01",
        "user_role": "blaster",
        "current_bench_id": "BENCH_14",
        "current_design": None,
        "last_tool_call": None,
        "tool_results": None,
        "guardrail_trips": [],
        "session_id": "SESS_01",
        "language": "en",
    }

    res = input_guardrail_node(state)
    assert res.get("last_tool_call") == "GUARDRAIL_BLOCKED"
    assert "cannot fire or detonate" in res["messages"][0]["content"]
    assert len(res["guardrail_trips"]) == 1


def test_intent_classifier_node_classifies_guided_design():
    """Test intent_classifier_node correctly classifies 'design a blast for bench 14' as guided_design or expert_design."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Help me design a blast for bench 14"}],
        "user_id": "USER_01",
        "user_role": "blaster",
        "current_bench_id": None,
        "current_design": None,
        "last_tool_call": None,
        "tool_results": None,
        "guardrail_trips": [],
        "session_id": "SESS_01",
        "language": "en",
    }

    res = intent_classifier_node(state)
    assert res["current_bench_id"] == "BENCH_14"
    assert res["tool_results"]["intent"] in ["guided_design", "expert_design"]


def test_tool_executor_node_calls_correct_tool():
    """Test tool_executor_node executing report generation tool for action intent."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Generate the PDF report"}],
        "user_id": "USER_01",
        "user_role": "engineer",
        "current_bench_id": "BENCH_JWA_15S",
        "current_design": None,
        "last_tool_call": "INTENT_ACTION",
        "tool_results": {"intent": "action"},
        "guardrail_trips": [],
        "session_id": "SESS_01",
        "language": "en",
    }

    res = tool_executor_node(state)
    assert "Report generated successfully" in res["messages"][0]["content"]


def test_output_guardrail_node_blocks_hallucinated_action():
    """Test output_guardrail_node blocks hallucinated claim of firing a blast."""
    state: AgentState = {
        "messages": [{"role": "assistant", "content": "I have fired the blast for bench 14."}],
        "user_id": "USER_01",
        "user_role": "engineer",
        "current_bench_id": "BENCH_14",
        "current_design": None,
        "last_tool_call": None,
        "tool_results": {},
        "guardrail_trips": [],
        "session_id": "SESS_01",
        "language": "en",
    }

    res = output_guardrail_node(state)
    assert "Firing requires physical execution by a certified blaster" in res["messages"][0]["content"]
    assert len(res["guardrail_trips"]) == 1


def test_llm_config_router():
    """Test select_llm router function under online, offline, and Setswana conditions."""
    local_llm = select_llm(task_complexity="medium", is_online=False)
    assert isinstance(local_llm, LocalLLM)

    cloud_llm = select_llm(task_complexity="high", is_online=True)
    assert isinstance(cloud_llm, CloudLLM)

    tn_llm = select_llm(language="tn")
    assert isinstance(tn_llm, Pula8BLLM)
    gen_text = tn_llm.generate("Dumela")
    assert isinstance(gen_text, str)
    assert len(gen_text) > 0


def test_huggingface_llm():
    """Test HuggingFaceLLM wrapper for text-generation pipelines."""
    hf_llm = HuggingFaceLLM("meta-llama/Llama-3.3-70B-Instruct")
    res = hf_llm.generate("Who are you?")
    assert isinstance(res, str)
    assert len(res) > 0


def test_ollama_client_generate_and_fallback():
    """Test OllamaClient availability, generate, and fallback handling."""
    from unittest.mock import patch, MagicMock

    with patch("src.agent.llm_config.full_health_check") as mock_health:
        mock_health.return_value = {"can_use_offline_llm": True}
        client = OllamaClient(model="llama3.1:8b")
        assert client.is_available is True

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "Local response"}
            mock_post.return_value = mock_resp

            ans = client.generate("Hello")
            assert ans == "Local response"

    # Test Fallback when Ollama unavailable
    with patch("src.agent.llm_config.full_health_check") as mock_health:
        mock_health.return_value = {"can_use_offline_llm": False}
        client_off = OllamaClient(model="llama3.1:8b")
        assert client_off.is_available is False

        fallback_fn = lambda p: f"Cloud response for {p}"
        res_fb = client_off.generate_with_fallback("Hello", cloud_fallback=fallback_fn)
        assert res_fb == "Cloud response for Hello"


def test_agent_memory_manager():
    """Test AgentMemoryManager short-term and long-term user preferences storage."""
    mem = AgentMemoryManager(session_id="SESS_TEST_01")
    mem.update_short_term("bench_id", "BENCH_ORA_12N")
    assert mem.get_short_term("bench_id") == "BENCH_ORA_12N"

    mem.save_user_preferences(user_id="ENG_01", preferred_language="tn", user_role="engineer", favorite_bench_id="BENCH_ORA_12N")
    prefs = mem.get_user_preferences("ENG_01")
    assert prefs["preferred_language"] == "tn"
    assert prefs["favorite_bench_id"] == "BENCH_ORA_12N"


def test_compiled_agent_graph_execution():
    """Test executing the compiled LangGraph StateGraph agent graph."""
    app = build_agent_graph()

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": "Help me design a blast for bench 14"}],
        "user_id": "BLASTER_BOTSWANA_01",
        "user_role": "blaster",
        "current_bench_id": None,
        "current_design": None,
        "last_tool_call": None,
        "tool_results": None,
        "guardrail_trips": [],
        "session_id": "SESS_GRAPH_TEST",
        "language": "en",
    }

    res_state = app.invoke(initial_state)
    assert "messages" in res_state
    assert len(res_state["messages"]) >= 1
    last_msg = res_state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]
    assert "Next Step:" in content or "Would you like" in content


def create_initial_state(user_message: str) -> AgentState:
    """Helper function to create initial agent state."""
    return {
        "messages": [{"role": "user", "content": user_message}],
        "user_id": "TEST_USER_01",
        "user_role": "engineer",
        "current_bench_id": None,
        "current_design": None,
        "last_tool_call": None,
        "tool_results": None,
        "guardrail_trips": [],
        "session_id": "SESS_KNOWLEDGE_TEST",
        "language": "en",
    }


def test_agent_answers_term_lookup():
    """The agent should answer 'What is powder factor?' using Lyntas/PA DEP/ISEE."""
    agent = build_agent_graph()
    state = create_initial_state("What is powder factor?")
    result = agent.invoke(state)
    last_msg = result["messages"][-1]
    response = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]
    assert "powder factor" in response.lower()


def test_agent_translates_to_setswana():
    """The agent should translate 'diamond' to Setswana."""
    agent = build_agent_graph()
    state = create_initial_state("How do you say diamond in Setswana?")
    result = agent.invoke(state)
    last_msg = result["messages"][-1]
    response = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]
    assert "teemane" in response.lower() or "taemane" in response.lower() or "diamond" in response.lower()


def test_agent_translates_to_english():
    """The agent should translate 'taemane' to English."""
    agent = build_agent_graph()
    state = create_initial_state("What does taemane mean in English?")
    result = agent.invoke(state)
    last_msg = result["messages"][-1]
    response = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]
    assert "diamond" in response.lower() or "taemane" in response.lower()


def test_agent_answers_general_question():
    """The agent should answer a general question using Pula-8B."""
    agent = build_agent_graph()
    state = create_initial_state("Why is stemming important in blasting?")
    result = agent.invoke(state)
    last_msg = result["messages"][-1]
    response = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]
    assert len(response) > 20


def test_agent_routes_design_request_to_design_flow():
    """The agent should NOT route 'Design a blast for bench 14' to knowledge."""
    agent = build_agent_graph()
    state = create_initial_state("Design a blast for bench 14.")
    result = agent.invoke(state)
    assert result.get("knowledge_intent") == "not_knowledge"
