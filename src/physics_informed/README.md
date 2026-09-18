# 🧠 BlastOpt Botswana - Physics-Informed Neural Network (`physics_informed`)

## 📌 Overview

The `physics_informed` package embeds fundamental drilling and blasting domain physics (Kuz-Ram rock fragmentation and USBM wave attenuation equations) directly into the GA-ANN loss function as differentiable soft constraints:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1 \cdot \mathcal{L}_{\text{kuzram}} + \lambda_2 \cdot \mathcal{L}_{\text{usbm}}$$

### Why Physics-Informed ML Matters:
Purely data-driven neural networks fail when predicting outside their training range (e.g. powder factor $> 1.20\text{ kg/m}^3$ or bench height $> 18\text{ m}$) because they lack awareness of physical laws. By embedding differentiable physics equations:
1. **Extrapolation Autonomy:** The model maintains monotonic physical behavior on out-of-distribution inputs.
2. **Data Efficiency:** Soft physics penalties accelerate convergence even when training data is sparse.
3. **Safety Guarantee:** Prevents unphysical predictions (e.g. negative fragmentation or non-attenuating vibration).

---

## 🏗️ Package Architecture

```text
src/physics_informed/
├── __init__.py            # Package exports
├── physics_equations.py   # Differentiable Kuz-Ram ($X_{50}$) & USBM ($PPV$) PyTorch & NumPy functions
├── pinn_model.py          # PhysicsInformedGAANN PyTorch neural network class
├── loss.py                # Composite physics loss function ($\mathcal{L}_{\text{total}}$)
├── trainer.py             # PINNTrainer with gradient clipping & dynamic loss weighting
├── evaluator.py           # In-distribution & OOD extrapolation evaluation
├── visualizer.py          # Plotly loss curves & extrapolation comparison charts
└── README.md              # Module guide & GA-ANN integration instructions
```

---

## 🚀 Quickstart & Usage Example

```python
import numpy as np
from src.physics_informed import PINNTrainer, PINNEvaluator

# 1. Initialize PINN Trainer with Physics Soft Penalties
trainer = PINNTrainer(
    epochs=150,
    batch_size=32,
    lambda_kuzram=0.25,
    lambda_usbm=0.25
)

# 2. Train on Blast Features (X) & Targets (Y)
# X columns: [burden, spacing, diameter, bench_h, stemming, subdrill, PF, charge, rock_A, RMR, dist, RWS]
history = trainer.train(X_train, Y_train)
print("Final Composite Loss:", history["final_loss"])

# 3. Evaluate Out-Of-Distribution (OOD) Extrapolation
evaluator = PINNEvaluator(model=trainer.model)
extrap_eval = evaluator.evaluate_extrapolation(ood_x, target_metric="powder_factor_kg_m3")

print("D50 Physics MAE:", extrap_eval["d50_physics_mae_mm"], "mm")
print("Monotonic Physics Alignment:", extrap_eval["obeys_physics_monotonicity"])
```

---

## 🔬 GA-ANN Model Integration Note

### How to Integrate `PhysicsInformedGAANN` into Existing Prediction Pipelines (`src/predict.py` or `src/models.py`):

In `src/models.py` or `src/predict.py`:

```python
from src.physics_informed import PhysicsInformedGAANN, PINNTrainer

class GAANNModelPipeline:
    def __init__(self):
        self.pinn_trainer = PINNTrainer(lambda_kuzram=0.25, lambda_usbm=0.25)

    def train_physics_informed(self, X: np.ndarray, Y: np.ndarray):
        """Trains GA-ANN with embedded Kuz-Ram and USBM soft loss constraints."""
        return self.pinn_trainer.train(X, Y)

    def predict_with_physics(self, input_features: np.ndarray) -> np.ndarray:
        """Predicts blast outcomes with physics-bounded extrapolation."""
        return self.pinn_trainer.model(input_features)
```

---

## ⚡ Running Demo & Tests

```bash
# Run interactive demo script
python3 demo_physics_informed.py

# Run unit tests
python3 -m pytest tests/test_physics_informed.py
```
