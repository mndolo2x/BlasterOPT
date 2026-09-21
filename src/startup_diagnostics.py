"""
Startup Diagnostics Subsystem for BlastOpt Botswana.

Safely inspects and verifies all core domain, service, ML, integration, and agent subsystems
at app startup without crashing or failing the application.
"""

import sys
import logging
import importlib
import traceback
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# List of all subsystems to inspect at startup
SUBSYSTEM_MODULES = [
    "src.config",
    "src.synthetic_data",
    "src.data_ingestion",
    "src.models",
    "src.predict",
    "src.optimize",
    "src.visualize",
    "src.report",
    "src.explainability",
    "src.recommender",
    "src.mwd_ingestion",
    "src.realtime_adaptive",
    "src.digital_twin",
    "src.drill_connectivity",
    "src.detonator_integration",
    "src.offline_sync",
    "src.regulatory",
    "src.i18n",
    "src.integrations",
    "src.pinn",
    "src.pareto_optimizer",
    "src.model_cards",
    "src.explainability_audit",
    "src.ensemble_uncertainty",
    "src.autshumato_translator",
    "src.domain.safety_checks",
    "src.domain.approval",
    "src.domain.blast_design",
    "src.services.approval_service",
    "src.services.prediction_service",
    "src.services.audit_service",
    "src.services.override_service",
    "src.agent.guardrails",
    "src.agent.voice_interface",
    "src.agent.agent_ui",
    "src.agent.ollama_health",
    "src.agent.audit",
]


def run_startup_diagnostics() -> Dict[str, Any]:
    """
    Executes a complete startup health check across all registered subsystems.

    Returns:
    --------
    Dict[str, Any]
        Structured diagnostic report dictionary containing overall status, counts,
        and per-module status breakdowns with captured tracebacks for any failures.
    """
    module_statuses: Dict[str, Dict[str, Any]] = {}
    passed_count = 0
    failed_count = 0

    for mod_name in SUBSYSTEM_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            module_statuses[mod_name] = {
                "status": "operational",
                "error": None,
                "traceback": None,
                "file": getattr(mod, "__file__", "built-in"),
            }
            passed_count += 1
        except Exception as err:
            tb_str = traceback.format_exc()
            logger.warning(f"Startup diagnostic check failed for module '{mod_name}': {err}")
            module_statuses[mod_name] = {
                "status": "failed",
                "error": str(err),
                "traceback": tb_str,
                "file": None,
            }
            failed_count += 1

    total_count = len(SUBSYSTEM_MODULES)

    if failed_count == 0:
        overall_status = "healthy"
    elif passed_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "critical"

    return {
        "overall_status": overall_status,
        "modules_checked_count": total_count,
        "modules_passed_count": passed_count,
        "modules_failed_count": failed_count,
        "module_statuses": module_statuses,
    }
