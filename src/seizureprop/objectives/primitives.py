"""Signal localization objective primitives."""
import torch
from torch import Tensor
from torch.nn import functional as F

def balanced_binary(logits: Tensor, target: Tensor, focal_gamma: float = 0) -> Tensor:
    """Class-balanced stable BCE; optionally focus on poorly classified examples."""
    target = target.expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if focal_gamma:
        probability_correct = torch.exp(-loss)
        loss = (1 - probability_correct).pow(focal_gamma) * loss
    parts = [loss[target == value].mean() for value in (0, 1) if (target == value).any()]
    return torch.stack(parts).mean()

def pairwise_rank(logits: Tensor, labels: Tensor) -> Tensor:
    """Encourage marked channels to outrank unmarked channels within one recording."""
    pos, neg = logits[labels > .5], logits[labels <= .5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.sum() * 0
    return F.softplus(neg[:, None] - pos[None, :]).mean()

def bag_logits(ictal: Tensor) -> Tensor:
    """Top-20%-window pooling: weak involvement label need not hold at all times."""
    return ictal.topk(max(1, ictal.shape[1] // 5), dim=1).values.mean(1)
