"""
Geology Domain Adaptation Package (`domain_adaptation`).
Handles model adaptation when transitioning from Kimberlite to Granite geologies
via layer-freezing transfer learning and adversarial feature alignment.
"""

from src.domain_adaptation.models import (
    DomainAdaptationConfig,
    FineTuneMetrics,
    DomainAdaptationReport,
)
from src.domain_adaptation.data_manager import DomainDataManager
from src.domain_adaptation.fine_tuner import TransferFineTuner
from src.domain_adaptation.dann import DomainAdversarialGAANN, DANNTrainer
from src.domain_adaptation.manager import DomainAdaptationManager
from src.domain_adaptation.visualizer import (
    plot_domain_feature_distribution,
    plot_adaptation_r2_comparison,
)

__all__ = [
    "DomainAdaptationConfig",
    "FineTuneMetrics",
    "DomainAdaptationReport",
    "DomainDataManager",
    "TransferFineTuner",
    "DomainAdversarialGAANN",
    "DANNTrainer",
    "DomainAdaptationManager",
    "plot_domain_feature_distribution",
    "plot_adaptation_r2_comparison",
]
