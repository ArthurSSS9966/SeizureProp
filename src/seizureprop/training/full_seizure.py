"""Supervision at its actual temporal resolution, without fabricated negatives."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def balanced_binary_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Average each observed class; negative targets denote unknown labels."""
    if logits.shape != target.shape:
        raise ValueError("Logits and targets must have identical shapes")
    known = target >= 0
    if not known.any():
        raise ValueError("No observed labels")
    y, z = target[known], logits[known]
    if not ((y == 0) | (y == 1)).all():
        raise ValueError("Observed labels must be binary")
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return torch.stack([bce[y == k].mean() for k in (0, 1) if (y == k).any()]).mean()


def activity_loss(logits: Tensor, onset: Tensor, at_ten: Tensor | None,
                  baseline_frames: int = 10) -> tuple[Tensor, dict[str, float]]:
    """Learn one activity task from baseline, sparse snapshots and positive bags.

    The ictal bag spans the COMPLETE available seizure. An onset-positive or
    ten-second-positive channel is known to participate somewhere, but is not
    labeled positive at every second. All other ictal channel bags stay unknown.
    No ten-second label is propagated to the remainder of the seizure.
    """
    if logits.ndim != 2 or not 0 < baseline_frames < logits.shape[1]:
        raise ValueError("Baseline and ictal samples are required")
    if onset.shape != logits.shape[:1]:
        raise ValueError("Channel target shape mismatch")
    baseline = F.binary_cross_entropy_with_logits(
        logits[:, :baseline_frames], torch.zeros_like(logits[:, :baseline_frames]))
    positive = onset > 0
    terms = {"baseline": baseline}
    if at_ten is not None:
        if logits.shape[1] < baseline_frames + 10 or at_ten.shape != onset.shape:
            raise ValueError("Ten-second annotation is outside available signal")
        terms["snapshot"] = balanced_binary_loss(logits[:, baseline_frames + 9], at_ten)
        positive = positive | (at_ten > 0)
    if positive.any():
        # Positive-only multiple-instance supervision: at least one ictal frame.
        bag = logits[positive, baseline_frames:].amax(dim=1)
        terms["positive_bag"] = F.binary_cross_entropy_with_logits(bag, torch.ones_like(bag))
    return torch.stack(list(terms.values())).mean(), {k: float(v.detach()) for k, v in terms.items()}


def covering_intervals(start: int, stop: int, length: int) -> list[tuple[int, int]]:
    """Partition all frames, including a final short interval without dropping it."""
    if start < 0 or stop <= start or length < 1:
        raise ValueError("Invalid coverage interval")
    return [(i, min(i + length, stop)) for i in range(start, stop, length)]
