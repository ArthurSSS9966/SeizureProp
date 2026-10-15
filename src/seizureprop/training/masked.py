"""Baseline-only masked reconstruction with disjoint identity calibration."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch.nn import functional as F


def reconstruction_mask(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw a reproducible mask with at least one hidden and visible token."""
    if x.ndim != 3 or x.shape[0] * x.shape[-1] < 2:
        raise ValueError("Reconstruction requires at least two channel-time tokens")
    mask = torch.rand((x.shape[0], x.shape[-1]), generator=generator) < .5
    # Preserve existing draws unless they would make the objective undefined.
    if not mask.any() or mask.all():
        index = int(torch.randint(mask.numel(), (), generator=generator))
        mask.reshape(-1)[index] = not bool(mask.all())
    return mask.to(x.device)


def masked_objective(model: torch.nn.Module, x: torch.Tensor,
                     calibration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Score only hidden absolute features; targets cannot enter the encoder."""
    if not mask.any() or mask.all():
        raise ValueError("Reconstruction needs both hidden and visible tokens")
    predicted = model.reconstruct(x, calibration, mask)
    target = x[:, :predicted.shape[1], :]
    return F.smooth_l1_loss(predicted.permute(0, 2, 1)[mask],
                            target.permute(0, 2, 1)[mask])


def pretrain(model: torch.nn.Module, training: list[dict], validation: list[dict],
             seed: int, epochs: int = 40, patience: int = 10) -> tuple[dict, list[dict], int]:
    """Train on external training patients; choose epoch on fixed validation masks.

    Records expose only patient, x and calibration. Neither onset nor tissue
    labels are used. Masks and recording draws use dedicated random generators.
    """
    generator = torch.Generator().manual_seed(seed + 101)
    rng = np.random.default_rng(seed + 101)
    groups: dict[str, list[dict]] = {}
    for record in training:
        groups.setdefault(record['patient'], []).append(record)
    if set(groups) & {r['patient'] for r in validation}:
        raise ValueError("Pretraining and selection patients must be disjoint")
    for r in training + validation:
        x = r['x']
        if x.ndim != 3 or x.shape[0] * x.shape[-1] < 2:
            raise ValueError("Reconstruction requires at least two channel-time tokens")
    fixed_masks = [reconstruction_mask(r['x'], generator) for r in validation]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, best_epoch, best_state, history = float('inf'), -1, None, []
    patients = sorted(groups)
    for epoch in range(epochs + 1):
        losses = []
        if epoch:
            model.train()
            for index in rng.permutation(len(patients)):
                records = groups[patients[index]]
                r = records[rng.integers(len(records))]
                mask = reconstruction_mask(r['x'], generator)
                loss = masked_objective(model, r['x'], r['calibration'], mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        per_patient: dict[str, list[float]] = {}
        with torch.no_grad():
            for r, mask in zip(validation, fixed_masks):
                value = float(masked_objective(model, r['x'], r['calibration'], mask))
                per_patient.setdefault(r['patient'], []).append(value)
        metric = float(np.mean([np.mean(v) for v in per_patient.values()]))
        if not np.isfinite(metric):
            raise FloatingPointError("Nonfinite reconstruction loss")
        history.append({'epoch': epoch, 'validation_loss': metric,
                        'training_loss': float(np.mean(losses)) if losses else 0.})
        if metric < best - 1e-7:
            best, best_epoch, best_state = metric, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return best_state, history, best_epoch
