"""
Agent State Definition for BlasterOPT LangGraph Agent.

Defines the AgentState TypedDict used across LangGraph conversation graph nodes.
"""

from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    user_id: str
    user_role: str  # supervisor, engineer, blaster, etc.
    current_bench_id: Optional[str]
    current_design: Optional[dict]  # BlastDesign as dict
    last_tool_call: Optional[str]
    tool_results: Optional[dict]
    guardrail_trips: List[dict]
    session_id: str
    language: str  # "en" or "tn"
    knowledge_intent: Optional[str]
    knowledge_direction: Optional[str]
    knowledge_term: Optional[str]
