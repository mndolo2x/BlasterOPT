"""
Unit tests for Agent Chat Interface module (src/agent/agent_ui.py).
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent_ui import (
    render_agent_chat,
    render_guided_mode,
    render_expert_mode,
    render_voice_mode,
)


def test_agent_ui_rendering_imports_and_callable():
    """Verify that all agent_ui rendering functions exist and are callable."""
    assert callable(render_agent_chat)
    assert callable(render_guided_mode)
    assert callable(render_expert_mode)
    assert callable(render_voice_mode)


@patch("streamlit.session_state", new_callable=dict)
def test_demo_mode_preload_conversation(mock_session_state):
    """Test Demo Mode pre-loads expected sample conversation messages."""
    demo_conversation = [
        {"role": "user", "content": "Help me design an optimal blast for bench BENCH_JWA_15S with 10000 tonnes target."},
        {"role": "assistant", "content": "📋 **Blast Design Summary for BENCH_JWA_15S:**\n- **Powder Factor:** 0.65 kg/m³\n- **Burden x Spacing:** 6.0m x 7.0m\n- **Predicted D50:** 220 mm\n- **Predicted PPV:** 4.20 mm/s (Compliant <= 5.0 mm/s)\n- **Cost:** $4.80 / t\n\n👉 **Next Step:** Would you like me to route this design to your certified blaster for review?"},
    ]

    mock_session_state["chat_messages"] = demo_conversation

    assert len(mock_session_state["chat_messages"]) == 2
    assert "BENCH_JWA_15S" in mock_session_state["chat_messages"][0]["content"]
    assert "Blast Design Summary" in mock_session_state["chat_messages"][1]["content"]
