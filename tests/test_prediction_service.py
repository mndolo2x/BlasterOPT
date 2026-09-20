"""
Unit tests for PredictionService and pure predictions (src/domain/predictions.py & src/services/prediction_service.py).
"""

import pytest
from src.domain.predictions import BlastPrediction, PredictionValue
from src.services.prediction_service import PredictionService, RecommendationResult


def test_pure_prediction_no_recommendations():
    service = PredictionService()
    design = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 450.0,
    }

    pred = service.predict_with_uncertainty(design)

    assert isinstance(pred, BlastPrediction)
    assert pred.ppv_mms.mean > 0.0
    assert pred.ppv_mms.upper_95 >= pred.ppv_mms.mean
    assert not hasattr(pred, "recommendation")


def test_recommendation_safe_design():
    service = PredictionService()
    design = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 200.0,  # Low charge per delay -> low PPV
        "monitoring_distance_m": 1000.0,   # Large distance
    }
    constraints = {"max_ppv_mm_s": 15.0, "max_airblast_db": 130.0, "flyrock_exclusion_zone_m": 300.0}

    res = service.recommend(design, constraints=constraints)

    assert isinstance(res, RecommendationResult)
    assert res.status == "SAFE"
    assert res.confidence == "HIGH"
    assert res.recommendation == design
    assert res.safety_report is not None
    assert res.blocks_export is False


def test_recommendation_blocked_when_unsafe():
    service = PredictionService()
    design = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 1500.0,
        "max_charge_per_delay_kg": 3000.0, # Huge charge per delay -> high PPV > max limit
        "monitoring_distance_m": 50.0,      # Close distance -> unsafe PPV
    }
    constraints = {"max_ppv_mm_s": 2.0} # Strict 2.0 mm/s limit -> UNSAFE

    res = service.recommend(design, constraints=constraints)

    assert isinstance(res, RecommendationResult)
    assert res.status == "UNSAFE"
    assert res.recommendation is None  # MUST return None for unsafe design
    assert res.confidence == "NONE - UNSAFE DESIGN"
    assert res.safety_report is not None
    assert res.blocks_export is True
    assert res.requires_engineer_review is True


def test_recommendation_requires_review_flag():
    service = PredictionService()
    # At monitoring_distance_m=500, mean PPV is 9.63 mm/s.
    # Setting max_ppv_mm_s=10.0 gives mean (9.63) <= 10.0 < U95 (11.36)
    design = {
        "rock_factor_A": 8.0,
        "bench_height_m": 12.0,
        "hole_diameter_mm": 250.0,
        "burden_m": 6.0,
        "spacing_m": 7.0,
        "stemming_m": 5.0,
        "powder_factor_kg_m3": 0.65,
        "charge_mass_per_hole_kg": 320.0,
        "max_charge_per_delay_kg": 640.0,
        "monitoring_distance_m": 500.0,
    }
    constraints = {"max_ppv_mm_s": 10.0}

    res = service.recommend(design, constraints=constraints)

    assert isinstance(res, RecommendationResult)
    assert res.status == "REQUIRES_REVIEW"
    assert res.confidence == "REQUIRES ENGINEER REVIEW"
    assert res.recommendation == design
    assert res.safety_report is not None
    assert res.blocks_export is False
    assert res.requires_engineer_review is True
