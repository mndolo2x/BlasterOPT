"""
Category definitions for hierarchical navigation in BlastOpt Botswana.
Defines 7 top-level categories and 25 sub-pages.
"""

from typing import Any, Dict

CATEGORIES: Dict[str, Dict[str, Any]] = {
    "analytics": {
        "title": "Analytics & Data",
        "icon": "📊",
        "description": "Explore historical blast records and KPIs",
        "pages": [
            {"file": "pages/dashboard.py", "title": "Dashboard & Data Explorer", "icon": "📊", "description": "KPIs and raw historical blast data"},
            {"file": "pages/visualize.py", "title": "Visualize", "icon": "📈", "description": "Custom plots and sensitivity analysis"},
        ],
    },
    "models": {
        "title": "Models & Training",
        "icon": "🧠",
        "description": "Train, compare, and document the AI models",
        "pages": [
            {"file": "pages/data_ingestion.py", "title": "Data Ingestion & Generator", "icon": "📥", "description": "Generate synthetic or upload real blast data"},
            {"file": "pages/ml_model_manager.py", "title": "ML Model Manager", "icon": "🧠", "description": "Train and save GA-ANN, PINN, Ensemble UQ"},
            {"file": "pages/model_comparison.py", "title": "Model Comparison", "icon": "⚖️", "description": "Compare R², RMSE, MAE across models"},
            {"file": "pages/model_cards.py", "title": "Model Cards & Audit", "icon": "📋", "description": "Documentation for every deployed model"},
            {"file": "pages/debswana_integrations.py", "title": "Debswana Integrations", "icon": "🔗", "description": "Connect to SAP, Deswik, Surpac"},
        ],
    },
    "design": {
        "title": "Blast Design",
        "icon": "🎯",
        "description": "Design a complete blast bench",
        "pages": [
            {"file": "pages/conversational_agent.py", "title": "Conversational Agent", "icon": "🗣️", "description": "Plain-language blast design (English & Setswana)"},
            {"file": "pages/blast_pattern.py", "title": "2D Blast Pattern & Delays", "icon": "📐", "description": "Visual layout of holes and timing sequence"},
            {"file": "pages/blast_pattern_3d.py", "title": "3D Blast Pattern Design", "icon": "🧊", "description": "3D pattern layout, bench face, and deck design"},
            {"file": "pages/blast_3d_viewer.py", "title": "3D Blast Engineering Viewer", "icon": "🔍", "description": "8-tab 3D bench, subdrill, toe burden, backbreak, collision & timing analysis"},
            {"file": "pages/timing_design.py", "title": "Advanced Timing Design", "icon": "⏱️", "description": "Electronic detonator initiation patterns and delay optimization"},
            {"file": "pages/digital_twin.py", "title": "Digital Twin of Bench", "icon": "💎", "description": "3D visualization with what-if scenarios"},
            {"file": "pages/predictor.py", "title": "Predictor & Kuz-Ram Curve", "icon": "📈", "description": "Predict fragmentation, PPV, airblast"},
            {"file": "pages/pinn.py", "title": "PINN Prediction & Uncertainty", "icon": "🧠", "description": "Physics-informed predictions with 95% CI"},
            {"file": "pages/uncertainty.py", "title": "Uncertainty Quantification", "icon": "📉", "description": "Aleatoric vs. epistemic decomposition"},
        ],
    },
    "optimization": {
        "title": "Optimization & Cost",
        "icon": "⚡",
        "description": "Find the best trade-off between cost, fragmentation, and vibration",
        "pages": [
            {"file": "pages/ga_optimizer.py", "title": "Genetic Algorithm Optimizer", "icon": "⚡", "description": "Optimize burden, spacing, powder factor"},
            {"file": "pages/pareto_optimizer.py", "title": "Multi-Objective Pareto Optimizer", "icon": "⚖️", "description": "5-objective trade-off designs"},
            {"file": "pages/economic_dashboard.py", "title": "Economic Dashboard", "icon": "💰", "description": "Cost per tonne, crusher throughput"},
            {"file": "pages/similar_blasts.py", "title": "Similar Blasts Recommender", "icon": "👥", "description": "Find past blasts with similar parameters"},
        ],
    },
    "compliance": {
        "title": "Compliance & Audit",
        "icon": "🛡️",
        "description": "Regulatory checks, safety guardrails, and immutable logs",
        "pages": [
            {"file": "pages/regulatory_compliance.py", "title": "Regulatory Compliance", "icon": "📋", "description": "Automated Botswana PPV/airblast checks"},
            {"file": "pages/guardrail_log.py", "title": "Guardrail Log", "icon": "🛡️", "description": "Every safety refusal, logged"},
            {"file": "pages/agent_audit_log.py", "title": "Agent Audit Log", "icon": "📝", "description": "Every AI interaction, immutably logged"},
        ],
    },
    "execution": {
        "title": "Field Execution",
        "icon": "🚜",
        "description": "Push designs to drills, MMUs, and detonators",
        "pages": [
            {"file": "pages/drill_connectivity.py", "title": "Drill Connectivity", "icon": "🚛", "description": "Push design to Sandvik/Epiroc rigs"},
            {"file": "pages/mwd_monitoring.py", "title": "Real-Time MWD Monitoring", "icon": "📡", "description": "Live Measure-While-Drilling data"},
            {"file": "pages/detonator_integration.py", "title": "Electronic Detonator Integration", "icon": "⚡", "description": "Upload timing to AEL, BME, Orica"},
            {"file": "pages/sync_status.py", "title": "Sync Status & Write-Ahead Log", "icon": "🔄", "description": "Offline-first sync for remote mines"},
        ],
    },
    "system": {
        "title": "System Health",
        "icon": "⚙️",
        "description": "System diagnostics and status",
        "pages": [
            {"file": "pages/ollama_health.py", "title": "Ollama System Health", "icon": "🔧", "description": "Offline LLM status and model check"},
        ],
    },
}
