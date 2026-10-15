"""Separate optimization of waveform channel experts with sparse labels."""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from ..models.channel_experts import ChannelExpert, masked_channel_bce
from .waveform_study import FitConfig


def fit_channel_expert(
    model: ChannelExpert,
    patients: list[str],
    sample: Callable[[str, np.random.Generator], tuple[Tensor, Tensor, Tensor | None]],
    validate: Callable[[ChannelExpert], float],
    config: FitConfig,
    seed: int,
) -> tuple[list[dict], int]:
    """Fit one task; select its epoch using validation-patient average precision."""
    if not patients or config.epochs < 0:
        raise ValueError("Nonempty training patients and nonnegative epochs required")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best, epoch_best = -float("inf"), 0
    selected = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(config.epochs + 1):
        losses = []
        if epoch:
            model.train()
            for index in rng.permutation(len(patients)):
                signal, target, auxiliary = sample(patients[index], rng)
                loss = masked_channel_bce(model(signal, auxiliary), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite channel objective")
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
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None,
                        "validation_AP": metric})
        if metric > best:
            best, epoch_best, selected = metric, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(selected)
    return history, epoch_best
