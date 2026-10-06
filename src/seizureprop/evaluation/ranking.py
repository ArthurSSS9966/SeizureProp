"""Tie-aware channel retrieval with the legacy denominator convention."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

def ranking_metrics(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Tie-aware MRP and top-20 expected recall, with standard AP/AUC."""
    positive = y > .5
    if not positive.any() or positive.all():
        raise ValueError("Ranking requires labeled and unmarked channels")
    ranks = rankdata(-scores, method="average") - 1
    k = min(20, len(scores))
    threshold = np.sort(scores)[-k]
    above, tied = scores > threshold, scores == threshold
    recall = (positive[above].sum() + positive[tied].sum() * (k - above.sum()) / tied.sum()) / positive.sum()
    return {"AP": float(average_precision_score(y, scores)),
            "AUC": float(roc_auc_score(y, scores)),
            "MRP": float(100 * np.median(ranks[positive]) / (len(y) - 1)),
            "Hits20": float(recall)}

def expected_topk(scores: np.ndarray, k: int = 20) -> np.ndarray:
    """Fractional inclusion weights at ties; independent of array/contact order."""
    k = min(k, len(scores))
    threshold = np.sort(scores)[-k]
    weights = (scores > threshold).astype(float)
    tied = scores == threshold
    weights[tied] = (k - weights.sum()) / tied.sum()
    return weights

def retrieval(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    values = ranking_metrics(scores, y)
    n = int(y.sum())
    values["AccAtN"] = float(np.dot(expected_topk(scores, n), y) / n)
    return values
