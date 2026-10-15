"""Baseline identity pretraining with consistency and optional tissue supervision.

Variance/covariance regularization follows the VICReg principle
(https://arxiv.org/abs/2105.04906). This small channel-embedding adaptation has
no projector and is not a reproduction of its vision architecture.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def baseline_views(x: Tensor, nbase: int) -> tuple[Tensor, Tensor]:
    """Return disjoint twelve-second views, with a gap and onset guard.

    Input is channels by features by one-second frames. The last two baseline
    frames are excluded. No ictal frame is read, even when provided in x.
    """
    if x.ndim != 3 or not 28 <= nbase <= x.shape[-1]:
        raise ValueError("At least 28 baseline frames required")
    return x[:, :, :12], x[:, :, nbase - 14:nbase - 2]


def consistency_terms(first: Tensor, second: Tensor) -> dict[str, Tensor]:
    """Align matched channels while penalizing collapse and redundant dimensions."""
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2:
        raise ValueError("Matched channel-by-embedding arrays with two channels required")
    invariance = F.mse_loss(first, second)
    variance = first.new_zeros(())
    covariance = first.new_zeros(())
    for values in (first, second):
        centered = values - values.mean(0)
        variance = variance + F.relu(1 - torch.sqrt(values.var(0, unbiased=False) + 1e-4)).mean() / 2
        cov = centered.T @ centered / (len(values) - 1)
        off_diagonal = cov - torch.diag_embed(cov.diagonal())
        covariance = covariance + off_diagonal.square().sum() / values.shape[1]
    # The conventional 25:25:1 ratio, divided by 25 for interpretable weighting.
    total = invariance + variance + .04 * covariance
    return {'consistency': total, 'invariance': invariance,
            'variance': variance, 'covariance': covariance}


def tissue_objective(logits: Tensor, labels: Tensor) -> Tensor:
    """Class-balance verified pure tissue pairs; -1 is excluded, never negative."""
    if logits.shape != labels.shape or not torch.isin(labels, labels.new_tensor([-1, 0, 1])).all():
        raise ValueError("Tissue targets must be aligned -1/0/1 labels")
    terms = [F.binary_cross_entropy_with_logits(logits[labels == label],
             torch.full_like(logits[labels == label], float(label)))
             for label in (0, 1) if (labels == label).any()]
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def identity_objective(model: nn.Module, head: nn.Module, record: dict,
                       consistency_weight: float, tissue_weight: float) -> tuple[Tensor, dict[str, float]]:
    """Compute pretraining loss; tissue is an optional target, never an input."""
    first = model.channel_identity(record['first'])
    second = model.channel_identity(record['second'])
    terms = consistency_terms(first, second)
    tissue = first.sum() * 0
    if tissue_weight:
        labels = record['tissue']
        tissue = (tissue_objective(head(first).squeeze(-1), labels)
                  + tissue_objective(head(second).squeeze(-1), labels)) / 2
    loss = consistency_weight * terms['consistency'] + tissue_weight * tissue
    terms.update(tissue=tissue, loss=loss,
                 embedding_std=(first.std(0, unbiased=False).mean()
                                + second.std(0, unbiased=False).mean()) / 2)
    return loss, {name: float(value.detach()) for name, value in terms.items()}


@torch.no_grad()
def identity_metrics(model: nn.Module, head: nn.Module, records: list[dict],
                     consistency_weight: float, tissue_weight: float) -> dict[str, float]:
    """Average recordings within patient before averaging patients."""
    model.eval()
    head.eval()
    patients: dict[str, list[dict]] = {}
    for r in records:
        _, metrics = identity_objective(model, head, r, consistency_weight, tissue_weight)
        patients.setdefault(r['patient'], []).append(metrics)
    if not patients:
        raise ValueError("Empty validation cohort")
    return {key: float(np.mean([np.mean([r[key] for r in rows])
                               for rows in patients.values()]))
            for key in next(iter(patients.values()))[0]}


def pretrain_identity(model: nn.Module, training: list[dict], validation: list[dict],
                      seed: int, consistency_weight: float = 1., tissue_weight: float = 0.,
                      epochs: int = 40, patience: int = 10) -> tuple[dict, dict, list[dict], int]:
    """Fit only identity layers and an auxiliary head, preserving other weights.

    Training and validation must contain different patients. Views are baseline
    only. Validation chooses the epoch using the same fixed pretraining loss;
    neither clinical onset nor any held-out localization score is consulted.
    """
    if min(consistency_weight, tissue_weight) < 0 or consistency_weight + tissue_weight <= 0:
        raise ValueError("At least one nonnegative pretraining weight is required")
    if not training or not validation or epochs < 0 or patience < 1:
        raise ValueError("Nonempty partitions and valid epoch limits required")
    if {r['patient'] for r in training} & {r['patient'] for r in validation}:
        raise ValueError("Training and validation patients overlap")
    if not model.use_identity:
        raise ValueError("Identity must be enabled")
    if tissue_weight and any(not any((r['tissue'] >= 0).any() for r in records)
                             for records in (training, validation)):
        raise ValueError("Tissue supervision requires known labels in each partition")
    rng = np.random.default_rng(seed + 202)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 202)
        head = nn.Linear(model.embed.out_channels, 1)
    parameters = list(model.id_frames.parameters()) + list(model.id_pool.parameters())
    optimizer = torch.optim.AdamW(parameters + list(head.parameters()), lr=.001, weight_decay=.001)
    groups: dict[str, list[dict]] = {}
    for r in training:
        groups.setdefault(r['patient'], []).append(r)
    people = sorted(groups)
    best, best_epoch, state, head_state, history = np.inf, -1, None, None, []
    for epoch in range(epochs + 1):
        losses = []
        if epoch:
            model.train()
            head.train()
            for index in rng.permutation(len(people)):
                records = groups[people[index]]
                record = records[rng.integers(len(records))]
                loss, _ = identity_objective(model, head, record, consistency_weight, tissue_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite identity objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters + list(head.parameters()), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        metrics = identity_metrics(model, head, validation, consistency_weight, tissue_weight)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite validation diagnostic")
        history.append({'epoch': epoch, 'training_loss': float(np.mean(losses)) if losses else 0.,
                        **{f'validation_{name}': value for name, value in metrics.items()}})
        if metrics['loss'] < best - 1e-7:
            best, best_epoch = metrics['loss'], epoch
            state, head_state = copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(state)
    return state, head_state, history, best_epoch
