"""
Classifiers, Baselines, and Prediction Review Logic for GeoResponse MWD Rock-Type Classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from src.georesponse.config import CONFIDENCE_THRESHOLD

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """
    Dummy Majority-Class Baseline Classifier.
    Always predicts the most frequent class in the training split.
    """

    def __init__(self):
        self.majority_class_ = None
        self.classes_ = None

    def fit(self, X: Any, y: Any) -> "MajorityClassBaseline":
        y_arr = np.asarray(y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        self.majority_class_ = self.classes_[np.argmax(counts)]
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.majority_class_)

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        probas = np.zeros((n_samples, len(self.classes_)))
        idx = list(self.classes_).index(self.majority_class_)
        probas[:, idx] = 1.0
        return probas


class GeoResponseClassifier:
    """
    Dataset-aware wrapper for Random Forest, Majority Baseline, and optional CatBoost/LightGBM rock-type classifiers.
    Tracks fitted feature schema to validate prediction inputs and prevent target leakage.
    """

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.classes_ = None
        self.feature_names_: List[str] = []

        if self.model_type == "majority_baseline":
            self.model = MajorityClassBaseline()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state)
        elif self.model_type == "catboost" and HAS_CATBOOST:
            self.model = CatBoostClassifier(iterations=100, verbose=0, random_seed=random_state)
        elif self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = LGBMClassifier(n_estimators=100, random_state=random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state)

    def fit(self, X: Any, y: Any) -> "GeoResponseClassifier":
        # Record fitted feature schema
        if isinstance(X, pd.DataFrame):
            # Exclude target leakage or non-numeric metadata columns
            non_feat_cols = ["rock_type_label", "rock_class", "target", "tunnel_id", "hole_id", "blast_id", "timestamp"]
            feat_cols = [
                c for c in X.columns
                if c not in non_feat_cols and pd.api.types.is_numeric_dtype(X[c])
            ]
            self.feature_names_ = list(feat_cols)
            X_mat = X[self.feature_names_].fillna(X[self.feature_names_].median())
        else:
            X_mat = np.asarray(X)
            self.feature_names_ = [f"feat_{i}" for i in range(X_mat.shape[1])]

        self.model.fit(X_mat, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def _validate_input_schema(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing_cols = [c for c in self.feature_names_ if c not in X.columns]
            if missing_cols:
                raise ValueError(
                    f"Incompatible dataset schema during prediction. Model requires trained features: {self.feature_names_}, "
                    f"but dataset is missing required features: {missing_cols}."
                )
            return X[self.feature_names_].fillna(X[self.feature_names_].median())
        else:
            X_mat = np.asarray(X)
            if X_mat.shape[1] != len(self.feature_names_):
                raise ValueError(f"Feature matrix dimension mismatch: expected {len(self.feature_names_)} features, got {X_mat.shape[1]}.")
            return pd.DataFrame(X_mat, columns=self.feature_names_)

    def predict(self, X: Any) -> np.ndarray:
        X_mat = self._validate_input_schema(X)
        return self.model.predict(X_mat)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_mat = self._validate_input_schema(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_mat)
        preds = self.predict(X_mat)
        probas = np.zeros((len(X_mat), len(self.classes_)))
        for i, p in enumerate(preds):
            if p in self.classes_:
                c_idx = list(self.classes_).index(p)
                probas[i, c_idx] = 1.0
        return probas

    def predict_with_review_flags(self, X: Any) -> pd.DataFrame:
        """
        Generates predictions alongside class probabilities, confidence scores,
        and review warnings for low-confidence or uncertain predictions.
        """
        X_mat = self._validate_input_schema(X)
        preds = self.predict(X_mat)
        probas = self.predict_proba(X_mat)

        confidences = np.max(probas, axis=1)
        flags = [c < CONFIDENCE_THRESHOLD for c in confidences]

        results = []
        for i, (p, conf, flag) in enumerate(zip(preds, confidences, flags)):
            results.append({
                "predicted_rock_type": str(p),
                "confidence_score": float(round(conf, 4)),
                "requires_review_flag": bool(flag),
                "review_warning": "⚠️ Low Confidence (< 70%). Review measured core log before adjusting design." if flag else "✅ Confident Prediction",
            })

        return pd.DataFrame(results)
