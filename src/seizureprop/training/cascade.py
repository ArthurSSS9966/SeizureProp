"""Sparse ictal supervision and causal event gating for independent experts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from seizureprop.training.full_seizure import balanced_binary_loss


def ictal_participation_loss(logits: Tensor, onset: Tensor,
                             at_ten: Tensor | None = None) -> Tensor:
    """Supervise only known ictal labels; never invent whole-event negatives.

    The input starts at clinical onset and covers the complete seizure. The
    ten-second snapshot supplies both classes at its actual time. Known positive
    channels must participate somewhere in the event (positive-only MIL).
    """
    if logits.ndim != 2 or logits.shape[1] < 1 or onset.shape != logits.shape[:1]:
        raise ValueError("Invalid channel-time logits or onset targets")
    positive = onset > 0
    terms = []
    if at_ten is not None:
        if logits.shape[1] < 10 or at_ten.shape != onset.shape:
            raise ValueError("Ten-second target does not match signal")
        terms.append(balanced_binary_loss(logits[:, 9], at_ten))
        positive = positive | (at_ten > 0)
    if positive.any():
        z = logits[positive].amax(dim=1)
        terms.append(F.binary_cross_entropy_with_logits(z, torch.ones_like(z)))
    if not terms:
        raise ValueError("No observed participation labels")
    return torch.stack(terms).mean()


def onset_discrimination_loss(logits: Tensor, onset: Tensor,
                              at_ten: Tensor | None = None) -> Tensor:
    """Binary onset objective with explicit later-participant hard negatives.

    Give equal weight to onset positives and all negatives; within negatives,
    balance known later participants against the other onset-negative channels.
    This is still a single onset task, not a propagation ranking objective.
    """
    if logits.shape != onset.shape:
        raise ValueError("Onset target shape mismatch")
    if at_ten is not None and at_ten.shape != onset.shape:
        raise ValueError("Participation target shape mismatch")
    known = torch.isfinite(onset) & (onset >= 0)
    if ((onset[known] != 0) & (onset[known] != 1)).any():
        raise ValueError("Observed onset targets must be binary")
    # Slice before BCE: excluded NaNs must not poison backward on shared weights.
    logits, onset = logits[known], onset[known]
    at_ten = at_ten[known] if at_ten is not None else None
    positive, negative = onset > 0, onset == 0
    bce = F.binary_cross_entropy_with_logits(logits, onset, reduction="none")
    terms = [bce[positive].mean()] if positive.any() else []
    late = negative & (at_ten > 0) if at_ten is not None else torch.zeros_like(negative)
    negatives = [bce[mask].mean() for mask in (late, negative & ~late) if mask.any()]
    if negatives:
        terms.append(torch.stack(negatives).mean())
    if not terms:
        raise ValueError("No observed onset targets")
    return torch.stack(terms).mean()


@dataclass(frozen=True)
class Episode:
    """Frame indices: half-open data interval, plus causal alarm/close times."""

    start: int
    stop: int
    alarm: int
    estimated_start: int
    censored: bool


def detect_episodes(probability: np.ndarray, threshold: float,
                    persistence: int = 2, release: int = 5,
                    buffer: int = 30) -> list[Episode]:
    """Apply hysteresis without clinical boundaries or future look-ahead.

    Entry requires consecutive high frames; exit requires consecutive frames
    below half the entry threshold. Endpoints are available only at closure.
    The trailing release interval is retained, and open tails are censored.
    """
    p = np.asarray(probability)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected a finite probability vector")
    if not 0 < threshold < 1 or min(persistence, release) < 1 or buffer < 0:
        raise ValueError("Invalid event gate configuration")
    episodes, high, low, active = [], 0, 0, None
    for i, value in enumerate(p):
        if active is None:
            high = high + 1 if value >= threshold else 0
            if high >= persistence:
                first = i - persistence + 1
                active = (max(0, first-buffer), i, first)
                low = 0
        else:
            low = low + 1 if value < threshold / 2 else 0
            if low >= release:
                episodes.append(Episode(active[0], i+1, active[1], active[2], False))
                active, high, low = None, 0, 0
    if active is not None:
        episodes.append(Episode(active[0], len(p), active[1], active[2], True))
    return episodes
