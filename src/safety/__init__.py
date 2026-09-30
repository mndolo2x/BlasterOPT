"""
Safety & Environmental package for BlasterOPT / BlastOpt Botswana.
"""

from src.safety.dust import DustModel
from src.safety.dust_dispersion import predict_dust_dispersion
from src.safety.toxic_gases import GasModel, predict_toxic_gases
from src.safety.noise_prediction import NoiseModel, predict_noise_overpressure
from src.safety.environmental_impact import EnvironmentalAssessment, evaluate_environmental_impact
from src.safety.risk_analysis import BlastRiskAnalyzer, calculate_risk_matrix
from src.safety.renderer import (
    SafetyRenderer,
    plot_dust_dispersion,
    plot_gas_concentration,
    plot_noise_attenuation,
    plot_risk_matrix,
    plot_environmental_breakdown,
)

__all__ = [
    "DustModel",
    "GasModel",
    "NoiseModel",
    "EnvironmentalAssessment",
    "BlastRiskAnalyzer",
    "predict_dust_dispersion",
    "predict_toxic_gases",
    "predict_noise_overpressure",
    "evaluate_environmental_impact",
    "calculate_risk_matrix",
    "SafetyRenderer",
    "plot_dust_dispersion",
    "plot_gas_concentration",
    "plot_noise_attenuation",
    "plot_risk_matrix",
    "plot_environmental_breakdown",
]
