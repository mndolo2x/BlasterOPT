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
    Wrapper for Random Forest, Majority Baseline, and optional CatBoost/LightGBM rock-type classifiers.
    """

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.classes_ = None

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
        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        preds = self.predict(X)
        # Dummy one-hot probas
        probas = np.zeros((len(X), len(self.classes_)))
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
        preds = self.predict(X)
        probas = self.predict_proba(X)

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
