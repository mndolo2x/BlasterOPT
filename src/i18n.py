"""
Internationalization (i18n) Module for BlastOpt Botswana.

Provides English and Setswana translation dictionaries and string lookup helpers for the Streamlit web app.

Setswana Domain & Mining Terminology Context:
---------------------------------------------
Setswana is the national language of Botswana, spoken natively by over 90% of the population.
Providing Setswana language support in BlastOpt ensures local blasters, pit operators, drillers,
and mine technicians can operate software with confidence.

Mining Terminology Differences (English vs Setswana):
- Drilling & Blasting: "Thunyako le Go Tlhaba Garane" (Blasting and Hole Drilling)
- Powder Factor (kg/m3): "Sekala sa Dithunyane" (Explosives scale per volume)
- Ground Vibration (PPV): "Tshikinyego ya Mmu" (Earth shaking / vibration)
- Flyrock: "Mabopo a a Fofang" (Flying rock fragments)
- Fragmentation (d50): "Bogolo ba Mabu a a Thuntsheditsweng" (Fragmented rock size distribution)
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "BlastOpt Botswana",
        "app_subtitle": "AI-Driven Drilling & Blasting Design, Fragmentation Modeling & Genetic Algorithm Optimizer",
        "nav_dashboard": "📊 Dashboard & Data Explorer",
        "nav_ingestion": "⚙️ Data Ingestion & Generator",
        "nav_ml_manager": "🤖 ML Model Manager",
        "nav_comparison": "🔬 Model Comparison",
        "nav_predictor": "🎯 Predictor & Kuz-Ram Curve",
        "nav_optimizer": "⚡ Genetic Algorithm Optimizer",
        "nav_pareto": "⚡ Multi-Objective Pareto Optimizer",
        "nav_economic": "💰 Economic Dashboard",
        "nav_recommender": "👥 Similar Blasts Recommender",
        "nav_mwd": "📡 Real-Time MWD Monitoring",
        "nav_digital_twin": "💎 Digital Twin of Bench",
        "nav_connectivity": "🚜 Drill Connectivity",
        "nav_detonator": "⚡ Electronic Detonator Integration",
        "nav_sync": "🔄 Sync Status & Write-Ahead Log",
        "nav_regulatory": "📜 Regulatory Compliance",
        "nav_pinn": "🧠 PINN Prediction & Uncertainty",
        "nav_integrations": "🔗 Debswana Integrations",
        "nav_model_cards": "📋 Model Cards & Audit",
        "nav_ensemble_uq": "🛡️ Uncertainty Quantification",
        "nav_pattern": "📐 2D Blast Pattern & Delays",
        "nav_agent": "🤖 Conversational Agent",
        "nav_guardrail_log": "🛡️ Guardrail Log",
        "nav_audit_log": "📜 Agent Audit Log",
        "nav_visualize": "📈 Visualize",
        "btn_predict": "Predict Outcomes",
        "btn_optimize": "Run GA Optimization",
        "btn_download_report": "Download PDF Report",
        "label_powder_factor": "Powder Factor (kg/m3)",
        "label_burden": "Burden (m)",
        "label_spacing": "Spacing (m)",
        "label_stemming": "Stemming (m)",
        "label_vibration": "Ground Vibration (PPV)",
        "label_flyrock": "Flyrock Distance",
        "label_cost": "Total Cost per Tonne",
    },
    "tn": {
        "app_title": "BlastOpt Botswana (Motheo ba Setswana)",
        "app_subtitle": "Mekgwa ya Sebaletsi ya Go Thunya Mmu, Tekanyetso ya Ditshikinyego le Katiso ya Tikologo",
        "nav_dashboard": "📊 Tjhate le Go Batlisisa Ditlhaka",
        "nav_ingestion": "⚙️ Tsaiso le Tlhamo ya Ditlhaka",
        "nav_ml_manager": "🤖 Molaodi wa Didiriswa tsa ML",
        "nav_comparison": "🔬 Tshwantsho ya Didiriswa",
        "nav_predictor": "🎯 Seboledi le Tjhate ya Kuz-Ram",
        "nav_optimizer": "⚡ Katiso ya Algorithm ya GA",
        "nav_pareto": "⚡ Katiso ya Dintlha tse Dintsi tsa Pareto",
        "nav_economic": "💰 Tjhate ya Ditshenyegelo",
        "nav_recommender": "👥 Batlhagisi ba Dithunya tsa Bogologolo",
        "nav_mwd": "📡 Tekolo ya Nako ya MWD",
        "nav_digital_twin": "💎 Sebopeho sa 3D sa Bench",
        "nav_connectivity": "🚜 Kgokagano ya Metjhini ya Go Tlhaba",
        "nav_detonator": "⚡ Didiriswa tsa Dithunya tsa Motlakase",
        "nav_sync": "🔄 Maemo a Sync le Khomputara ya Offline",
        "nav_regulatory": "📜 Melao le Ditheko tsa Puso",
        "nav_pinn": "🧠 Setsebi sa PINN sa Sebaletsi",
        "nav_integrations": "🔗 Kgokagano tsa Debswana Enterprise",
        "nav_model_cards": "📋 Dibukana tsa Model le Haholo",
        "nav_ensemble_uq": "🛡️ Tekodiso ya Ditshikinyego le Tlhokomelo",
        "nav_pattern": "📐 Sebopeho sa Dithunya sa 2D",
        "nav_agent": "🤖 Sebuabuisane sa Sebaletsi (Agent)",
        "nav_guardrail_log": "🛡️ Khomputara ya Tlhokomelo (Guardrails)",
        "nav_audit_log": "📜 Pegelo ya Di-Audit tsa Agent",
        "nav_visualize": "📈 Pontsho ya Pitso",
        "btn_predict": "Bolela Diphetogo",
        "btn_optimize": "Diragatsa Katiso ya GA",
        "btn_download_report": "Gapa Pegelo ya PDF",
        "label_powder_factor": "Sekala sa Dithunyane (kg/m3)",
        "label_burden": "Sebaka sa Burden (m)",
        "label_spacing": "Sebaka sa Spacing (m)",
        "label_stemming": "Sebaka sa Stemming (m)",
        "label_vibration": "Tshikinyego ya Mmu (PPV)",
        "label_flyrock": "Mabopo a a Fofang (Flyrock)",
        "label_cost": "Ditshenyegelo ka Tonne",
    },
}


def get_translation(key: str, lang: str = "en") -> str:
    """
    Returns the localized translation string for a given key.

    Parameters:
    -----------
    key : str
        Translation key identifier.
    lang : str, default="en"
        Language code ("en" for English, "tn" for Setswana).

    Returns:
    --------
    str
        Localized translation string.
    """
    lang_clean = lang.lower().strip()
    if lang_clean not in TRANSLATIONS:
        lang_clean = "en"

    return TRANSLATIONS[lang_clean].get(key, TRANSLATIONS["en"].get(key, key))
