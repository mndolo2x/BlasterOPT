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
from src.agent.llm_config import select_llm, CloudLLM, LocalLLM
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
    """Test select_llm router function under online and offline conditions."""
    local_llm = select_llm(task_complexity="medium", is_online=False)
    assert isinstance(local_llm, LocalLLM)

    cloud_llm = select_llm(task_complexity="high", is_online=True)
    assert isinstance(cloud_llm, CloudLLM)


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
