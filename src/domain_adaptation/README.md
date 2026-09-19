# Geology Domain Adaptation Package (`src/domain_adaptation`)

## Overview
The `domain_adaptation` package manages the transition of machine learning blast optimization models from **Kimberlite** to **Granite** geologies. When transitioning to harder granite rock mass (higher Rock Factor $A$, lower $RMR$, higher compressive strength), zero-shot models trained on softer kimberlite suffer performance degradation.

This package implements a multi-stage domain adaptation framework:
1. **Transfer Learning Fine-Tuning:** Freezes early feature extraction layers and fine-tunes later regression heads using a small target granite dataset ($N \approx 30\text{--}50$).
2. **Domain-Adversarial Neural Network (DANN):** If fine-tuning target $R^2 < 0.75$, DANN uses a Gradient Reversal Layer (GRL) and domain classifier (Kimberlite=0 vs Granite=1) to learn domain-invariant feature representations.

---

## Architecture & Modules

* `models.py`: Pydantic data schemas (`DomainAdaptationConfig`, `FineTuneMetrics`, `DomainAdaptationReport`).
* `data_manager.py`: Source and target domain data synthesis and feature alignment (`DomainDataManager`).
* `fine_tuner.py`: PyTorch layer-freezing transfer fine-tuner (`TransferFineTuner`).
* `dann.py`: Domain-Adversarial Neural Network with Gradient Reversal Layer (`DomainAdversarialGAANN`, `DANNTrainer`).
* `manager.py`: Main orchestrator (`DomainAdaptationManager`).
* `visualizer.py`: Plotly PCA distribution alignment and pre- vs. post-adaptation $R^2$ bar charts.

---

## Quickstart Usage

```python
from src.domain_adaptation import DomainAdaptationConfig, DomainAdaptationManager, DomainDataManager
from src.physics_informed import PhysicsInformedGAANN

# 1. Initialize data manager & generate synthetic Kimberlite (Source) and Granite (Target) data
data_mgr = DomainDataManager()
X_src, Y_src, X_tgt, Y_tgt = data_mgr.create_synthetic_domain_datasets(n_source=200, n_target=40)

# 2. Base Kimberlite model
base_model = PhysicsInformedGAANN(input_dim=12, output_dim=4)

# 3. Adapt domain using DomainAdaptationManager
config = DomainAdaptationConfig(
    source_geology="Kimberlite",
    target_geology="Granite",
    freeze_early_layers=True,
    fine_tune_epochs=50,
    target_r2_threshold=0.75
)

adapter = DomainAdaptationManager(config=config)
adapted_model, report = adapter.adapt_domain(base_model, X_src, Y_src, X_tgt, Y_tgt)

print(f"Pre-adaptation R2: {report.metrics.pre_adaptation_r2}")
print(f"Post-adaptation R2: {report.metrics.post_adaptation_r2}")
print(f"Adaptation Method: {report.metrics.method_used}")
```
