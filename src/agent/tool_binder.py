"""
Tool Binder Module for BlasterOPT Conversational Agent.

Binds domain execution functions to TOOL_REGISTRY entries.
"""

import logging
from typing import Any, Dict
from src.agent.tool_registry import (
    TOOL_REGISTRY,
    ToolRegistry,
    _predict_fragmentation_handler,
    _predict_vibration_handler,
    _predict_airblast_handler,
    _predict_downstream_handler,
    _design_blast_handler,
    _optimize_blast_handler,
    _explain_prediction_handler,
    _get_mwd_data_handler,
    _get_geology_handler,
    _get_regulations_handler,
    _search_past_blasts_handler,
    _find_similar_blasts_handler,
    _generate_report_handler,
    _route_for_approval_handler,
    _log_decision_handler,
    _query_knowledge_graph_handler,
)

logger = logging.getLogger(__name__)


def bind_tools(registry: ToolRegistry = TOOL_REGISTRY) -> ToolRegistry:
    """
    Binds domain execution functions to all registered entries in TOOL_REGISTRY and returns the updated registry.
    """
    registry["predict_fragmentation"]["function"] = _predict_fragmentation_handler
    registry["predict_vibration"]["function"] = _predict_vibration_handler
    registry["predict_airblast"]["function"] = _predict_airblast_handler
    registry["predict_downstream"]["function"] = _predict_downstream_handler
    registry["design_blast"]["function"] = _design_blast_handler
    registry["optimize_blast"]["function"] = _optimize_blast_handler
    registry["explain_prediction"]["function"] = _explain_prediction_handler
    registry["get_mwd_data"]["function"] = _get_mwd_data_handler
    registry["get_geology"]["function"] = _get_geology_handler
    registry["get_regulations"]["function"] = _get_regulations_handler
    registry["search_past_blasts"]["function"] = _search_past_blasts_handler
    registry["find_similar_blasts"]["function"] = _find_similar_blasts_handler
    registry["generate_report"]["function"] = _generate_report_handler
    registry["route_for_approval"]["function"] = _route_for_approval_handler
    registry["log_decision"]["function"] = _log_decision_handler
    registry["query_knowledge_graph"]["function"] = _query_knowledge_graph_handler

    logger.info("Successfully bound functions to all TOOL_REGISTRY entries.")
    return registry
