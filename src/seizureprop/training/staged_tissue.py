"""Controlled staged versus auxiliary tissue learning on a common signal model.

Calibration occupies the first twelve baseline frames. Readouts used for loss
start after calibration; these interfaces do not implement continuous detection.
"""
from __future__ import annotations

import copy
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import torch
from scipy.signal import firwin, upfirdn
from torch import Tensor, nn

from ..models.channel_identity import ChannelIdentityScorer
from ..models.tissue import TissueClassifier
from ..objectives.primitives import balanced_binary, pairwise_rank
from .identity_pretraining import tissue_objective


def causal_resample(signal: np.ndarray, source_hz: float, target_hz: int = 256) -> np.ndarray:
    """Resample with an unshifted causal FIR, retaining its physical delay.

    Output sample k depends only on input samples at or before k/target_hz.
    No centered-filter delay compensation or tail padding is returned.
    """
    if signal.ndim != 2 or source_hz < target_hz or target_hz <= 0:
        raise ValueError('Expected time-by-channel signals and valid downsampling rates')
    ratio = Fraction(target_hz / source_hz).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    if abs(source_hz * up / down - target_hz) > 1e-8:
        raise ValueError('Sampling ratio cannot be represented precisely')
    if up == down:
        return signal.copy()
    width = max(up, down)
    taps = firwin(20 * width + 1, 1 / width, window=('kaiser', 5.0)) * up
    count = (len(signal) * up + down - 1) // down
    return upfirdn(taps, signal, up=up, down=down, axis=0)[:count]


def fit_tissue_stage(rows: pd.DataFrame, patients: set[str]) -> TissueClassifier:
    """Fit only known, spontaneous training rows aggregated per patient/pair."""
    selected = rows[rows.patient.isin(patients) & (rows.y >= 0) & ~rows.stimulated]
    grouped = []
    for (patient, pair), group in selected.groupby(['patient', 'pair']):
        if group.y.nunique() != 1:
            raise ValueError('Inconsistent tissue within a patient/pair')
        grouped.append((patient, int(group.y.iloc[0]), np.median(np.stack(group.x), axis=0)))
    if {r[0] for r in grouped} != patients:
        raise ValueError('Every fitting patient needs eligible spontaneous tissue data')
    model = TissueClassifier('logistic', 1.0).fit(
        np.stack([r[2] for r in grouped]), np.array([r[1] for r in grouped]),
        np.array([r[0] for r in grouped]))
    model.fit_patients = sorted(patients)
    return model


def crossfit_tissue(rows: pd.DataFrame, training: set[str], folds: list[list[str]]) -> tuple[np.ndarray, TissueClassifier, list[dict]]:
    """Generate patient-excluded stage-two inputs; fit final model on train only.

    Non-training rows may be validation or test: their labels never enter fitting.
    All labels for an excluded patient, including repeated events, are excluded.
    """
    held = [p for fold in folds for p in fold]
    if len(held) != len(set(held)) or set(held) != training or any(not f for f in folds):
        raise ValueError('Folds must partition the training patients exactly once')
    scores = np.full(len(rows), np.nan)
    lineage = []
    for fold in folds:
        fit = training - set(fold)
        model = fit_tissue_stage(rows, fit)
        mask = rows.patient.isin(fold).to_numpy()
        scores[mask] = model.predict(np.stack(rows.loc[mask, 'x']))
        lineage.append({'fit': sorted(fit), 'predict': sorted(fold)})
    final = fit_tissue_stage(rows, training)
    mask = ~rows.patient.isin(training).to_numpy()
    if mask.any():
        scores[mask] = final.predict(np.stack(rows.loc[mask, 'x']))
    if not np.isfinite(scores).all():
        raise ValueError('Incomplete cross-fitted scores')
    return scores, final, lineage


class StagedTissueScorer(ChannelIdentityScorer):
    """Same backbone for three arms; tissue labels are never inference inputs."""

    def __init__(self, features: int = 34, hidden: int = 32) -> None:
        super().__init__(features, hidden)
        self.tissue_head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, nbase: int, stage_scores: Tensor | None = None) -> dict[str, Tensor]:
        if nbase < 28 or nbase >= x.shape[-1]:
            raise ValueError('Twenty-eight baseline frames and an onset frame required')
        calibration = x[:, :, :12]
        h = self.encode(x, calibration)
        logits = self.localization_head(h).squeeze(1)
        onset = logits[:, nbase]
        if stage_scores is not None:
            if stage_scores.shape != onset.shape or not torch.isfinite(stage_scores).all():
                raise ValueError('Finite aligned channel tissue scores required')
            if ((stage_scores < 0) | (stage_scores > 1)).any():
                raise ValueError('Tissue scores must be in [0,1]')
            # A bounded prior changes onset ranking only, never removes channels.
            onset = onset + torch.log(.1 + .9 * stage_scores)
        tissue = self.tissue_head(self.channel_identity(calibration)).squeeze(1)
        return {'localization': logits, 'onset': onset, 'tissue': tissue}


def staged_objective(outputs: dict[str, Tensor], record: dict, auxiliary_weight: float) -> Tensor:
    """Protect clinical-positive WM; mask unknown tissue and unsupported spread."""
    logits, onset = outputs['localization'], outputs['onset']
    # Negative readouts follow calibration, separated by four frames.
    baseline = logits[:, 16:record['nbase'] - 2]
    loss = .5 * (balanced_binary(baseline, torch.zeros_like(baseline), 2.)
                 + balanced_binary(onset, record['y'], 2.))
    loss = loss + .2 * pairwise_rank(onset, record['y'])
    if record['y_spread'] is not None:
        if logits.shape[1] - record['nbase'] < 10:
            raise ValueError('Spread supervision exceeds available window')
        score = logits[:, record['nbase']:record['nbase'] + 10].mean(1)
        loss = loss + .5 * balanced_binary(score, record['y_spread'], 2.)
    if auxiliary_weight and not record['stimulated']:
        loss = loss + auxiliary_weight * tissue_objective(outputs['tissue'], record['tissue_target'])
    return loss


def train_staged(model: StagedTissueScorer, training: list[dict], validation: list[dict],
                 arm: str, seed: int, evaluate_fn, epochs: int = 20,
                 deadline: float | None = None) -> tuple[list[dict], int]:
    """Equal update schedule and validation-only checkpoint selection for all arms."""
    if arm not in {'A', 'B', 'C'} or epochs < 1:
        raise ValueError('Known arm and positive epoch budget required')
    if {r['patient'] for r in training} & {r['patient'] for r in validation}:
        raise ValueError('Patient leakage between fitting and selection')
    groups = {}
    for r in training:
        groups.setdefault(r['patient'], []).append(r)
    patients = sorted(groups)
    rng = np.random.default_rng(seed + 1000)
    torch.manual_seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best, selected_epoch, state = [], -np.inf, -1, None
    for epoch in range(epochs + 1):
        losses = []
        if epoch:
            model.train()
            for index in rng.permutation(len(patients)):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError('Fixed pilot wall-time budget exhausted')
                choices = groups[patients[index]]
                r = choices[rng.integers(len(choices))]
                out = model(r['x'], r['nbase'], r['stage_score'] if arm == 'B' else None)
                loss = staged_objective(out, r, .1 if arm == 'C' else 0.)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        rows, _ = evaluate_fn(model, validation, arm)
        metric = float(pd.DataFrame(rows).groupby('patient').onset_AP.mean().mean())
        if not np.isfinite(metric):
            raise ValueError('Invalid validation endpoint')
        history.append({'epoch': epoch, 'validation_AP': metric,
                        'loss': float(np.mean(losses)) if losses else 0.})
        if metric > best + 1e-7:
            best, selected_epoch, state = metric, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history, selected_epoch
