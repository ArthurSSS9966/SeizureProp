"""Standalone tissue estimator; patient grouping is handled before fitting."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def patient_class_weights(patients: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Equal patient contribution, then equal present classes within each patient."""
    weights = np.zeros(len(target), dtype=float)
    for patient in np.unique(patients):
        mask = patients == patient
        classes = np.unique(target[mask])
        for label in classes:
            group = mask & (target == label)
            weights[group] = 1 / (len(classes) * group.sum())
    return weights * len(weights) / weights.sum()


class TissueClassifier:
    """Fit 51 signal features and return GM scores (not calibrated certainty)."""

    allowed_labels = (0, 1)

    def __init__(self, kind: str = "logistic", c: float = 1.0) -> None:
        self.kind = kind
        self.c = c

    def fit(self, x: np.ndarray, y: np.ndarray, patients: np.ndarray) -> "TissueClassifier":
        if x.ndim != 2 or x.shape[1] != 51 or len(x) != len(y) or len(y) != len(patients):
            raise ValueError("Expected matching rows of 51 tissue features, labels and patients")
        if not np.isfinite(x).all() or tuple(np.unique(y)) != self.allowed_labels:
            raise ValueError(f"Finite features and all tissue classes {self.allowed_labels} required")
        weights = patient_class_weights(patients, y)
        self.scaler = StandardScaler().fit(x, sample_weight=weights)
        scaled = self.scaler.transform(x)
        if self.kind == "logistic":
            self.estimator = LogisticRegression(C=self.c, max_iter=3000, random_state=17)
        elif self.kind == "boosting":
            self.estimator = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05,
                max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.,
                early_stopping=False, random_state=17)
        else:
            raise ValueError("Unsupported tissue estimator")
        self.estimator.fit(scaled, y, sample_weight=weights)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x)[:, 1]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return scores in allowed_labels order, without calibration claims."""
        if x.ndim != 2 or x.shape[1] != 51 or not np.isfinite(x).all():
            raise ValueError("Finite 51-feature channel rows required")
        return self.estimator.predict_proba(self.scaler.transform(x))


class PairTissueClassifier(TissueClassifier):
    """Three-class estimator for pure WM, pure GM and known mixed pairs.

    This describes bipolar composition, not individual electrode contact tissue.
    Unknown anatomy is excluded by the caller, never learned as a mixed label.
    """

    allowed_labels = (0, 1, 2)


def retention_threshold(scores: np.ndarray, y: np.ndarray, patients: np.ndarray,
                        retention: float = .95) -> float:
    """Largest score cutoff retaining at least the specified macro GM fraction.

    Supply development predictions only. A channel is rejected when score < t;
    tied scores are retained. The cutoff is not selected using localization labels.
    """
    if not 0 < retention <= 1 or not np.isfinite(scores).all():
        raise ValueError("Finite scores and valid retention required")
    gm = y == 1
    if not gm.any():
        raise ValueError("No development GM labels")
    weights = np.zeros(len(scores))
    gm_patients = np.unique(patients[gm])
    for patient in gm_patients:
        mask = gm & (patients == patient)
        weights[mask] = 1 / (len(gm_patients) * mask.sum())
    values = np.unique(np.r_[0., scores])
    valid = [t for t in values if weights[scores >= t].sum() >= retention - 1e-12]
    return float(max(valid))
