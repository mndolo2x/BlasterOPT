"""
Unit tests for Agent State TypedDict module (src/agent/state.py).
"""

from src.agent.state import AgentState


def test_agent_state_construction():
    """Test instantiating AgentState dictionary with all required keys."""
    state: AgentState = {
        "messages": [],
        "user_id": "USER_001",
        "user_role": "engineer",
        "current_bench_id": "BENCH_JWA_15S",
        "current_design": {"design_id": "DES_001", "powder_factor": 0.65},
        "last_tool_call": "predict_fragmentation",
        "tool_results": {"d50_cm": 22.0},
        "guardrail_trips": [],
        "session_id": "SESS_123",
        "language": "en",
    }

    assert state["user_id"] == "USER_001"
    assert state["user_role"] == "engineer"
    assert state["current_bench_id"] == "BENCH_JWA_15S"
    assert state["current_design"]["design_id"] == "DES_001"
    assert state["last_tool_call"] == "predict_fragmentation"
    assert state["language"] == "en"
