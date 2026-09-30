"""
Fragmentation module package for BlasterOPT / BlastOpt Botswana.
"""

from src.fragmentation.kuz_ram import predict_kuz_ram
from src.fragmentation.swebrec import SwebrecModel, swebrec_cumulative_passing
from src.fragmentation.kco import KCOModel, predict_kco
from src.fragmentation.distributions import (
    rosin_rammler_curve,
    lognormal_curve,
    compare_distributions,
    plot_distribution_comparison,
    calculate_goodness_of_fit,
)
from src.fragmentation.image_analysis import ImageAnalysisImporter, parse_wipfrag_data
from src.fragmentation.calibration import FragmentationCalibrator, calibrate_rock_factor
from src.fragmentation.renderer import (
    FragmentationRenderer,
    plot_fragmentation_curve,
    plot_model_comparison,
    plot_calibration_scatter,
)

__all__ = [
    "predict_kuz_ram",
    "SwebrecModel",
    "swebrec_cumulative_passing",
    "KCOModel",
    "predict_kco",
    "rosin_rammler_curve",
    "lognormal_curve",
    "compare_distributions",
    "plot_distribution_comparison",
    "calculate_goodness_of_fit",
    "ImageAnalysisImporter",
    "parse_wipfrag_data",
    "FragmentationCalibrator",
    "calibrate_rock_factor",
    "FragmentationRenderer",
    "plot_fragmentation_curve",
    "plot_model_comparison",
    "plot_calibration_scatter",
]
