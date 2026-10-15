"""Training-only tissue contrast and independently mined onset hard negatives.

SupCon adapts https://arxiv.org/abs/2004.11362 to two baseline views with
class-balanced anchors. It is applied to the actual identity embedding, without
a disposable projector. Tissue annotations never enter the inference model.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .identity_pretraining import baseline_views
from .localization import evaluate, onset_objective, selection_metric
from ..objectives.primitives import balanced_binary


def tissue_contrast(first: Tensor, second: Tensor, labels: Tensor,
                    temperature: float = .1) -> Tensor:
    """Pull verified same-tissue baseline embeddings together across two views.

    Mixed, unknown and conflicting pairs must be encoded as -1. Each recording
    supplies both tissue classes; single-class recordings contribute zero.
    Positives include the other view of the same pair and other same-tissue
    pairs. Self-comparisons are excluded from numerator and denominator.
    """
    if first.ndim != 2 or first.shape != second.shape or labels.shape != first.shape[:1]:
        raise ValueError("Aligned channel embeddings and tissue labels required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive finite temperature required")
    if not torch.isin(labels, labels.new_tensor([-1, 0, 1])).all():
        raise ValueError("Tissue labels must be -1, 0, or 1")
    known = labels >= 0
    if not all((labels == k).any() for k in (0, 1)):
        return (first.sum() + second.sum()) * 0
    z = F.normalize(torch.cat([first[known], second[known]]), dim=1)
    y = labels[known].repeat(2)
    diagonal = torch.eye(len(y), device=z.device, dtype=torch.bool)
    similarity = z @ z.T / temperature
    denominator = similarity.masked_fill(diagonal, -torch.inf).logsumexp(1)
    positive = (y[:, None] == y[None, :]) & ~diagonal
    per_anchor = denominator - (similarity * positive).sum(1) / positive.sum(1)
    return torch.stack([per_anchor[y == k].mean() for k in (0, 1)]).mean()


def hard_wm_ranking(scores: Tensor, target: Tensor, tissue: Tensor,
                    mined: Tensor, margin: float = 1.) -> Tensor:
    """Rank clinical onset positives above mined, clinically negative pure WM.

    `mined` must originate from predictions on patients excluded from the
    mining model's fitting AND checkpoint selection. This function rechecks
    eligibility even if a caller supplies an overinclusive mask.
    """
    if scores.ndim != 1 or any(v.shape != scores.shape for v in (target, tissue, mined)):
        raise ValueError("Aligned per-channel arrays required")
    if mined.dtype != torch.bool or not math.isfinite(margin) or margin < 0:
        raise ValueError("Boolean mining mask and nonnegative finite margin required")
    if not torch.isin(target, target.new_tensor([0, 1])).all():
        raise ValueError("Clinical targets must be binary")
    if not torch.isin(tissue, tissue.new_tensor([-1, 0, 1])).all():
        raise ValueError("Tissue labels must be -1, 0, or 1")
    negative = (tissue == 0) & (target == 0) & mined
    positive = target == 1
    if not negative.any() or not positive.any():
        return scores.sum() * 0
    return F.softplus(margin + scores[negative][None, :] - scores[positive][:, None]).mean()


def joint_train(model: nn.Module, training: list[dict], validation: list[dict],
                config, contrast_weight: float = 0., hard_weight: float = 0.,
                temperature: float = .1, margin: float = 1.) -> tuple[dict, list[dict], int]:
    """Use the reference localization objective plus ongoing auxiliary losses.

    Sampling, optimizer, dropout RNG, clipping and patient-macro clinical AP
    selection match the reference patient-uniform trainer. Optional metadata
    is read only when its objective is enabled. Auxiliary encoders contain no
    dropout, so additional losses do not perturb reference RNG draws.
    """
    if any(not math.isfinite(w) or w < 0 for w in (contrast_weight, hard_weight)):
        raise ValueError("Finite nonnegative objective weights required")
    if not training or not validation or config.sampling != 'patient_uniform':
        raise ValueError("Nonempty patient-uniform training and validation required")
    if {r['patient'] for r in training} & {r['patient'] for r in validation}:
        raise ValueError("Training/selection patient overlap")
    if config.spread_weight and any(r['y_spread'] is None for r in training):
        raise ValueError("Missing spread supervision")
    groups = {}
    for r in training:
        groups.setdefault(r['patient'], []).append(r)
    patients = sorted(groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best, best_epoch, state, history = -np.inf, -1, None, []
    for epoch in range(config.epochs + 1):
        losses = []
        if epoch:
            model.train()
            for i in np.random.permutation(len(patients)):
                runs = groups[patients[i]]
                r = runs[np.random.randint(len(runs))]
                logits = model(r['x'], r['nbase'])['localization']
                main = onset_objective(logits, r['nbase'], r['y'], config.focal_gamma, config.rank_weight)
                if config.spread_weight:
                    main = main + config.spread_weight * balanced_binary(
                        logits[:, r['nbase']:r['nbase'] + 10].mean(1), r['y_spread'], config.focal_gamma)
                contrast, hard = main.new_zeros(()), main.new_zeros(())
                if contrast_weight:
                    a, b = baseline_views(r['x'], r['nbase'])
                    contrast = tissue_contrast(model.channel_identity(a), model.channel_identity(b),
                                               r['tissue_target'], temperature)
                if hard_weight:
                    hard = hard_wm_ranking(logits[:, r['nbase']], r['y'], r['tissue_target'],
                                           r['mined'], margin)
                loss = main + contrast_weight * contrast + hard_weight * hard
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append([float(v.detach()) for v in (loss, main, contrast, hard)])
        metric = selection_metric(evaluate(model, validation)[0], config.selection)
        means = np.mean(losses, axis=0) if losses else np.zeros(4)
        history.append({'epoch': epoch, 'validation_metric': metric,
                        **dict(zip(('loss', 'localization', 'contrast', 'hard'), means.tolist()))})
        if metric > best + 1e-7:
            best, best_epoch, state = metric, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config.patience:
            break
    if state is None:
        raise ValueError("No valid selected checkpoint")
    model.load_state_dict(state)
    return state, history, best_epoch
