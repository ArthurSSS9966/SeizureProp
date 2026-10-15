"""Separate, patient-balanced training stages for multichannel waveform models."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..models.spatiotemporal import SeparateTaskModel, SpectralReconstruction


@dataclass(frozen=True)
class FitConfig:
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    gradient_clip: float = 1.0
    mask_fraction: float = 0.25


def balanced_contact_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Equal weight for present label classes; no tissue or ranking penalties."""
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    terms = [loss[labels == k].mean() for k in (0, 1) if (labels == k).any()]
    return torch.stack(terms).mean()


def fit_pretraining(
    model: SpectralReconstruction, patients: list[str],
    sample: Callable[[str, np.random.Generator], Tensor],
    validate: Callable[[SpectralReconstruction], float],
    config: FitConfig, seed: int,
) -> tuple[list[dict], int]:
    """Patient-balanced masked spectral learning, selected on validation loss."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    history, best, best_epoch, state = [], float("inf"), 0, copy.deepcopy(model.state_dict())
    for epoch in range(config.epochs + 1):
        losses = []
        if epoch:
            model.train()
            for index in rng.permutation(len(patients)):
                signal = sample(patients[index], rng)
                mask = torch.as_tensor(rng.random(signal.shape[:2]) < config.mask_fraction,
                                       device=signal.device)
                if not mask.any():
                    mask[0, 0] = True
                loss = model.loss(signal, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reconstruction objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            metric = validate(model)
        if not np.isfinite(metric):
            raise ValueError("Nonfinite validation reconstruction error")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                        "validation_loss": metric})
        if metric < best:
            best, best_epoch, state = metric, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history, best_epoch


def fit_task(
    model: SeparateTaskModel, patients: list[str],
    sample: Callable[[str, np.random.Generator], list[tuple[Tensor, Tensor]]],
    validate: Callable[[SeparateTaskModel], float],
    config: FitConfig, seed: int,
) -> tuple[list[dict], int]:
    """Fit one downstream task without updating or supervising other heads."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best, best_epoch, state = [], -float("inf"), 0, copy.deepcopy(model.state_dict())
    for epoch in range(config.epochs + 1):
        losses = []
        if epoch:
            model.train()
            for index in rng.permutation(len(patients)):
                examples = sample(patients[index], rng)
                objectives = []
                for signal, target in examples:
                    logits = model(signal)
                    objectives.append(F.binary_cross_entropy_with_logits(logits, target)
                                      if model.task == "detection" else balanced_contact_loss(logits, target))
                loss = torch.stack(objectives).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite task objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            metric = validate(model)
        if not np.isfinite(metric):
            raise ValueError("Nonfinite validation AP")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                        "validation_AP": metric})
        if metric > best:
            best, best_epoch, state = metric, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history, best_epoch
