# BlastOpt Model Registry Architecture (`models/`)

## Overview
The `models/` directory provides a dynamic, auto-discovering Model Registry architecture for BlastOpt Botswana. Any ML or physics-informed model placed under `models/` (including subdirectories like `models/custom/` or `models/sklearn_wrappers/`) is automatically registered and made available in the Streamlit UI dropdown without modifying any other code files or restarting the server.

---

## How to Add a New Model

Adding a new blast prediction model to BlastOpt Botswana requires **only one step**:

### Step 1: Drop a new Python file in `models/custom/`

Create a new Python file (e.g., `models/custom/my_new_model.py`) and implement a class inheriting from `BaseBlastModel`:

```python
import pandas as pd
from models.base import BaseBlastModel, ModelMetadata

class MyNewBlastModel(BaseBlastModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="MyNewBlastModel", config=kwargs)
        # Initialize your underlying estimator or PyTorch neural network here

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="my_new_model",
            display_name="My New Custom Blast Model",
            model_type="custom",
            description="High-accuracy machine learning model for fragmentation and PPV prediction.",
            version="1.0.0",
            author="Mining Engineer / Data Scientist",
            input_features=["burden", "spacing", "powder_factor", "stemming", "rock_factor"],
            output_features=["fragmentation_p80", "ppv", "airblast"],
            supports_training=True,
            supports_uncertainty=False,
            supports_explainability=True,
            requires_gpu=False,
            tags=["custom", "neural_network"]
        )

    def fit(self, X, y, **kwargs):
        # Implement your training logic here
        self.is_fitted = True
        return self

    def predict(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Implement your prediction logic returning a pandas DataFrame or numpy array
        return pd.DataFrame([[220.0, 4.20, 110.0]] * len(X_df), columns=self.get_metadata().output_features, index=X_df.index)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)
```

**That's it!** On the next Streamlit page refresh or clicking **"🔄 Rescan Registry"** in the Model Registry Diagnostics tab, `My New Custom Blast Model` will automatically appear in the **Select Algorithm** dropdown.

---

## Directory Structure

```
models/
├── __init__.py               # Exports BaseBlastModel, ModelMetadata, get_registry
├── base.py                   # Abstract BaseClass & ModelMetadata Pydantic schema
├── registry.py               # Auto-discovery engine scanning models/ directory
├── configs/
│   └── model_registry.yaml   # Hyperparameter metadata defaults
├── sklearn_wrappers/         # Scikit-Learn model wrappers
│   ├── random_forest.py
│   ├── xgboost.py
│   └── ridge.py
└── custom/                   # Custom neural/physics model wrappers
    ├── ga_ann.py
    ├── pinn.py
    ├── ensemble.py
    └── site_calibration_model.py
```

---

## Architecture Features

1. **Auto-Discovery:** Scans `models/` recursively using `importlib` and `inspect` for all subclasses of `BaseBlastModel`.
2. **Graceful Error Handling:** Broken model files or missing dependencies are caught, logged, and surfaced in the **Model Registry Diagnostics** tab without crashing the application.
3. **Capability Badges:** Models declare features like `supports_uncertainty`, `supports_explainability`, or `requires_gpu`, which are dynamically rendered in Streamlit UI badges.
