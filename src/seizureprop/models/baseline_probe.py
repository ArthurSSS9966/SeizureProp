"""External-data baseline probes with train-only scaling and patient weighting."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .tissue import patient_class_weights


class BaselineProbe:
    """Binary tissue probe supporting explicitly named variable-width features."""

    def __init__(self, kind: str = "logistic", c: float = .1) -> None:
        self.kind, self.c = kind, c

    def fit(self, x: np.ndarray, y: np.ndarray, patients: np.ndarray) -> "BaselineProbe":
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(patients) or not np.isfinite(x).all():
            raise ValueError("Matching finite feature, label and patient rows required")
        if tuple(np.unique(y)) != (0, 1):
            raise ValueError("Both pure tissue classes required; exclude unknown labels")
        self.n_features = x.shape[1]
        weights = patient_class_weights(patients, y)
        self.scaler = StandardScaler().fit(x, sample_weight=weights)
        if self.kind == "logistic":
            self.model = LogisticRegression(C=self.c, max_iter=4000, random_state=17)
        elif self.kind == "boosting":
            self.model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                min_samples_leaf=20, learning_rate=.05, l2_regularization=1.,
                early_stopping=False, random_state=17)
        else:
            raise ValueError("Unsupported baseline probe")
        self.model.fit(self.scaler.transform(x), y, sample_weight=weights)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.n_features or not np.isfinite(x).all():
            raise ValueError("Finite features with the trained width required")
        return self.model.predict_proba(self.scaler.transform(x))[:, 1]


def pair_gm_score(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Conservative min-contact score; not a calibrated joint probability.

    Both contacts must be observed. Missing/invalid scores stay unknown (NaN),
    allowing the caller to retain uncovered pairs explicitly.
    """
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Matching contact-score shapes required")
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
    return np.where(valid, np.minimum(a, b), np.nan)
