"""
Geology Domain Adaptation Package (`domain_adaptation`).
Handles model adaptation when transitioning from Kimberlite to Granite geologies
via layer-freezing transfer learning, Joint Domain Adaptation (JDA), and DANN adversarial alignment.
"""

from src.domain_adaptation.models import (
    DomainData,
    AdaptationResult,
    DomainShiftReport,
    DomainAdaptationConfig,
)
from src.domain_adaptation.data_manager import DomainDataManager
from src.domain_adaptation.fine_tuner import TransferFineTuner
from src.domain_adaptation.jda_aligner import JDAAligner, compute_rbf_mmd
from src.domain_adaptation.dann_model import DomainAdversarialGAANN, DANNTrainer
from src.domain_adaptation.evaluator import CrossDomainEvaluator
from src.domain_adaptation.domain_manager import DomainAdaptationManager
from src.domain_adaptation.visualizer import (
    plot_domain_feature_distribution,
    plot_domain_classifier_accuracy,
    plot_adaptation_r2_comparison,
    plot_cross_domain_method_comparison,
)

__all__ = [
    "DomainData",
    "AdaptationResult",
    "DomainShiftReport",
    "DomainAdaptationConfig",
    "DomainDataManager",
    "TransferFineTuner",
    "JDAAligner",
    "compute_rbf_mmd",
    "DomainAdversarialGAANN",
    "DANNTrainer",
    "CrossDomainEvaluator",
    "DomainAdaptationManager",
    "plot_domain_feature_distribution",
    "plot_domain_classifier_accuracy",
    "plot_adaptation_r2_comparison",
    "plot_cross_domain_method_comparison",
]
