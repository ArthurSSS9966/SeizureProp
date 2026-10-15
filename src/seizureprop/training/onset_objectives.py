"""Isolated onset objectives using available contact annotations only."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def onset_objective(logits: Tensor, onset: Tensor, at_ten: Tensor,
                    agreement: Tensor, grouped: bool = False,
                    unanimous_only: bool = False) -> Tensor:
    """Balance positives against negatives, optionally balancing negative types.

    Onset negatives with a known positive ten-second annotation form one group;
    remaining known onset negatives form the other. Neither group supplies a
    recruitment timestamp. Unknown ten-second labels are -1 and stay in the
    general onset-negative group. Unanimous supervision ignores discordant
    labels and retains the original reference for all evaluation channels.
    """
    if not (logits.shape == onset.shape == at_ten.shape == agreement.shape):
        raise ValueError('Contact arrays must have identical shape')
    known = torch.isfinite(onset) & ((onset==0)|(onset==1))
    if unanimous_only:
        known &= ((agreement==0)&(onset==0)) | ((agreement==1)&(onset==1))
    loss = F.binary_cross_entropy_with_logits(logits, onset.nan_to_num(0).clamp(0,1), reduction='none')
    terms = []
    positive, negative = known & (onset==1), known & (onset==0)
    if positive.any(): terms.append(loss[positive].mean())
    if negative.any():
        groups = [negative & (at_ten==1), negative & (at_ten!=1)] if grouped else [negative]
        terms.append(torch.stack([loss[g].mean() for g in groups if g.any()]).mean())
    if not terms: raise ValueError('No usable onset supervision')
    return torch.stack(terms).mean()
