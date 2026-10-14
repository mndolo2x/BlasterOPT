"""
GeoResponse MWD Prediction UI Helper and Range Validation Module.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Exact feature names and order required by the GeoResponse rock classification model
REQUIRED_FEATURE_ORDER: List[str] = [
    "penetration_rate_m_hr",
    "percussion_pressure_bar",
    "feed_pressure_bar",
    "rotation_pressure_bar",
    "flushing_pressure_bar",
    "rotation_speed_rpm",
    "depth_m",
    "feed_percussion_ratio",
]

# Sensible default values from typical Norwegian tunnel drilling operations
DEFAULT_MWD_INPUTS: Dict[str, Any] = {
    "penetration_rate_m_hr": 35.0,
    "percussion_pressure_bar": 180.0,
    "feed_pressure_bar": 90.0,
    "rotation_pressure_bar": 95.0,
    "flushing_pressure_bar": 10.0,
    "rotation_speed_rpm": 180,
    "depth_m": 3.5,
    "feed_percussion_ratio": 0.50,
}

# Display labels for UI inputs
FIELD_DISPLAY_NAMES: Dict[str, str] = {
    "penetration_rate_m_hr": "Penetration Rate (m/hr)",
    "percussion_pressure_bar": "Percussion Pressure (bar)",
    "feed_pressure_bar": "Feed Pressure (bar)",
    "rotation_pressure_bar": "Rotation Pressure (bar)",
    "flushing_pressure_bar": "Flushing Pressure (bar)",
    "rotation_speed_rpm": "Rotation Speed (rpm)",
    "depth_m": "Depth (m)",
    "feed_percussion_ratio": "Feed Percussion Ratio",
}

# Physically reasonable ranges for Norwegian training telemetry
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "penetration_rate_m_hr": (5.0, 80.0),
    "percussion_pressure_bar": (100.0, 250.0),
    "feed_pressure_bar": (40.0, 150.0),
    "rotation_pressure_bar": (40.0, 160.0),
    "flushing_pressure_bar": (3.0, 20.0),
    "rotation_speed_rpm": (80.0, 300.0),
    "depth_m": (0.5, 10.0),
    "feed_percussion_ratio": (0.2, 0.9),
}


def validate_mwd_input_ranges(inputs: Dict[str, Any]) -> List[str]:
    """
    Validates input telemetry against typical training data ranges.

    Returns a list of warning messages for any out-of-range inputs without blocking prediction.
    """
    warnings: List[str] = []
    for feat_key, (min_val, max_val) in FEATURE_RANGES.items():
        val = inputs.get(feat_key)
        if val is not None:
            if val < min_val or val > max_val:
                label = FIELD_DISPLAY_NAMES.get(feat_key, feat_key)
                warnings.append(
                    f"Warning: {label} ({feat_key}) is outside the typical range for the training data. Predictions may be unreliable."
                )
    return warnings


def build_ordered_feature_dataframe(inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Constructs an input DataFrame using the exact feature order expected by the model.

    CRITICAL NOTE ON FEATURE ORDERING:
    Scikit-learn models (such as RandomForestClassifier) and underlying NumPy matrices
    rely strictly on positional column order during fit and predict operations.
    Passing feature columns in a different or arbitrary order (e.g. from Python dict keys
    or UI element placement) will align features to wrong model weights, causing incorrect
    predictions, degraded performance, or runtime schema errors.
    """
    missing_feats = [col for col in REQUIRED_FEATURE_ORDER if col not in inputs]
    if missing_feats:
        raise ValueError(
            f"Prediction requires all eight MWD inputs. Missing required inputs: {missing_feats}"
        )

    row = {col: inputs[col] for col in REQUIRED_FEATURE_ORDER}
    return pd.DataFrame([row])[REQUIRED_FEATURE_ORDER]


def get_or_create_default_classifier():
    """
    Returns a default trained GeoResponseClassifier trained on synthetic MWD data
    if no model was explicitly trained during the session.
    """
    from src.georesponse.models import GeoResponseClassifier

    clf = GeoResponseClassifier(model_type="random_forest")
    # Generate representative MWD sample data covering all 8 features
    np.random.seed(42)
    n = 100
    df_synthetic = pd.DataFrame({
        "penetration_rate_m_hr": np.random.uniform(15.0, 60.0, size=n),
        "percussion_pressure_bar": np.random.uniform(120.0, 220.0, size=n),
        "feed_pressure_bar": np.random.uniform(50.0, 130.0, size=n),
        "rotation_pressure_bar": np.random.uniform(50.0, 140.0, size=n),
        "flushing_pressure_bar": np.random.uniform(4.0, 18.0, size=n),
        "rotation_speed_rpm": np.random.uniform(100.0, 250.0, size=n),
        "depth_m": np.random.uniform(1.0, 8.0, size=n),
        "feed_percussion_ratio": np.random.uniform(0.3, 0.8, size=n),
        "rock_type_label": np.random.choice(["Soft Rock", "Medium Rock", "Hard Rock"], size=n),
    })
    X = df_synthetic[REQUIRED_FEATURE_ORDER]
    y = df_synthetic["rock_type_label"]
    clf.fit(X, y)
    return clf
