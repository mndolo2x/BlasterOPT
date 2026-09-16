"""
Voice Interface Orchestrator Module for BlasterOPT Speech Layer.

Bridges speech-to-text recognition, AgentState graph execution, and text-to-speech synthesis into a seamless voice turn.
"""

import uuid
import logging
from typing import Dict, Any, Optional

from src.agent.speech_to_text import transcribe_audio, detect_language
from src.agent.text_to_speech import synthesize_speech
from src.agent.agent_graph import build_agent_graph
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# Global compiled Agent Graph instance
_AGENT_APP = None


def _get_agent_app():
    global _AGENT_APP
    if _AGENT_APP is None:
        _AGENT_APP = build_agent_graph()
    return _AGENT_APP


def start_voice_session(user_id: str = "USER_DEFAULT") -> str:
    """
    Initializes a new voice conversation session ID.
    """
    session_id = f"VOICE_SESS_{user_id}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Initialized new voice session: {session_id}")
    return session_id


def process_voice_turn(
    audio_bytes: bytes,
    user_id: str = "USER_DEFAULT",
    session_id: Optional[str] = None,
    user_role: str = "engineer",
) -> Dict[str, Any]:
    """
    Processes a complete voice turn:
    1. Detects language ('en' or 'tn').
    2. Transcribes audio bytes to text.
    3. Executes Agent State Graph conversation turn.
    4. Synthesizes response text into speech audio bytes.
    5. Returns dict with 'text', 'audio', and 'language'.
    """
    if session_id is None:
        session_id = start_voice_session(user_id=user_id)

    # 1. Detect language
    lang = detect_language(audio_bytes)

    # 2. Transcribe audio to text
    transcription = transcribe_audio(audio_bytes, language=lang)

    # 3. Pass text to agent graph
    app = _get_agent_app()
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": transcription}],
        "user_id": user_id,
        "user_role": user_role,
        "current_bench_id": None,
        "current_design": None,
        "last_tool_call": None,
        "tool_results": None,
        "guardrail_trips": [],
        "session_id": session_id,
        "language": lang,
    }

    graph_res = app.invoke(initial_state)

    messages = graph_res.get("messages", [])
    if messages:
        last_msg = messages[-1]
        response_text = last_msg.content if hasattr(last_msg, "content") else (last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg))
    else:
        response_text = "I have processed your request."

    # 4. Synthesize text response to speech audio bytes
    audio_response = synthesize_speech(text=response_text, language=lang)

    # 5. Return dict
    return {
        "text": response_text,
        "transcription": transcription,
        "audio": audio_response,
        "language": lang,
        "session_id": session_id,
    }
