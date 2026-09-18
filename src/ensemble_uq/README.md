# 🛡️ BlastOpt Botswana - GA-ANN Ensemble Uncertainty Quantification (`ensemble_uq`)

## 📌 Overview

The `ensemble_uq` package equips BlastOpt Botswana's GA-ANN blast prediction model with epistemic uncertainty quantification and Out-Of-Distribution (OOD) risk detection.

When predictions are requested outside training range (e.g. powder factor $> 1.20\text{ kg/m}^3$ or bench height $> 18\text{ m}$), point-prediction models produce unreliable estimates without warning. This module:
1. Trains an ensemble of $N=7$ GA-ANN models using distinct seeds, varied neural network architectures, and bootstrap sampling.
2. Computes ensemble mean, standard deviation, and 95% confidence intervals ($\text{mean} \pm 1.96 \cdot \sigma$).
3. Decomposes total uncertainty into **aleatoric** (data noise) and **epistemic** (model knowledge gap) variance.
4. Detects OOD extrapolation using **Mahalanobis distance**, **Isolation Forest**, and explicit domain range checks.

---

## 🏗️ Package Architecture

```text
src/ensemble_uq/
├── __init__.py                # Package exports
├── models.py                  # Pydantic data models (EnsemblePrediction, OODReport, TrainingConfig)
├── ensemble_trainer.py        # Train N=7 GA-ANN models across bootstrap data splits
├── ood_detector.py            # Mahalanobis distance, Isolation Forest & domain range checks
├── uncertainty_quantifier.py  # 95% CI bounds & aleatoric vs epistemic variance decomposition
├── prediction.py              # Main UncertaintyAwarePredictor class
├── visualizer.py              # Plotly interval, decomposition, OOD, & calibration charts
└── README.md                  # Package guide & GA-ANN integration instructions
```

---

## 🚀 Quickstart & Usage Example

```python
from src.ensemble_uq import UncertaintyAwarePredictor

# 1. Initialize Predictor
predictor = UncertaintyAwarePredictor(model_dir="models/ensemble_uq/")

# 2. Define Input Blast Parameters
blast_params = {
    "burden_m": 6.0,
    "spacing_m": 7.0,
    "bench_height_m": 12.0,
    "powder_factor_kg_m3": 0.65,
    "max_charge_per_delay_kg": 640.0,
    "monitoring_distance_m": 450.0
}

# 3. Predict with Uncertainty & OOD Check
pred = predictor.predict(blast_params)

print("Predicted Fragmentation D50 Mean:", pred.mean["d50_mm"], "mm")
print("95% CI Bounds:", (pred.lower_95["d50_mm"], pred.upper_95["d50_mm"]))
print("Is Out-Of-Distribution (OOD):", pred.is_ood)

# 4. Generate Natural Language Explanation
print("\nExplanation:", predictor.explain_uncertainty(pred))
```

---

## 🔬 GA-ANN Model Integration Note

### How to Integrate `UncertaintyAwarePredictor` into Existing Prediction Pipelines (`src/predict.py`):

In `src/predict.py` or inside the GA-ANN optimization cost function:

```python
from src.ensemble_uq import UncertaintyAwarePredictor

_ENSEMBLE_PREDICTOR = UncertaintyAwarePredictor()

def predict_single_blast_with_uncertainty(input_dict: dict) -> dict:
    """
    Wrapper substituting single point predictions with ensemble predictions + OOD checks.
    """
    ensemble_res = _ENSEMBLE_PREDICTOR.predict(input_dict)

    if ensemble_res.is_ood:
        logger.warning(f"OOD Warning triggered: {ensemble_res.ood_reason}")

    return {
        "d50_mm": ensemble_res.mean.get("d50_mm", 220.0),
        "d50_mm_ci_95": (ensemble_res.lower_95.get("d50_mm", 200.0), ensemble_res.upper_95.get("d50_mm", 240.0)),
        "ppv_mms": ensemble_res.mean.get("ppv_mms", 4.2),
        "ppv_mms_ci_95": (ensemble_res.lower_95.get("ppv_mms", 3.5), ensemble_res.upper_95.get("ppv_mms", 4.9)),
        "is_ood": ensemble_res.is_ood,
        "explanation": _ENSEMBLE_PREDICTOR.explain_uncertainty(ensemble_res)
    }
```

---

## ⚡ Running Demo & Tests

```bash
# Run interactive demo script
python3 demo_ensemble_uq.py

# Run unit tests
python3 -m pytest tests/test_ensemble_uq.py
```
