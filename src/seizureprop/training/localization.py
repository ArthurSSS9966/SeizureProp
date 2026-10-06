"""Clinical onset training, optional spread supervision, patient-level selection."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import torch
from ..evaluation.ranking import retrieval, expected_topk
from ..objectives.primitives import balanced_binary, pairwise_rank


def fit_scaler(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([r['features'].reshape(-1, r['features'].shape[-1]) for r in records])
    return values.mean(0), np.maximum(values.std(0), .1)


def tensor_records(records: list[dict], mean: np.ndarray, scale: np.ndarray, device: str) -> list[dict]:
    return [{**r, 'x': torch.tensor(np.clip((r['features'] - mean) / scale, -12, 12).transpose(1, 2, 0), device=device),
             'y': torch.tensor(r['target'], device=device),
             'y_spread': None if r['spread'] is None else torch.tensor(r['spread'], device=device)} for r in records]


def onset_objective(logits: torch.Tensor, nbase: int, target: torch.Tensor, gamma: float, rank_weight: float) -> torch.Tensor:
    baseline = balanced_binary(logits[:, :nbase], torch.zeros_like(logits[:, :nbase]), gamma)
    onset = balanced_binary(logits[:, nbase], target, gamma)
    return .5 * (baseline + onset) + rank_weight * pairwise_rank(logits[:, nbase], target)


@torch.no_grad()
def evaluate(model: torch.nn.Module, records: list[dict], include_anatomy: bool = False) -> tuple[list[dict], list[dict]]:
    model.eval()
    rows, predictions = [], []
    for r in records:
        logits = model(r['x'], r['nbase'])['localization'].cpu().numpy()
        scores = logits[:, r['nbase']]
        row = {'patient': r['patient'], 'recording': r['recording']}
        row.update({f'onset_{k}': v for k, v in retrieval(scores, r['target']).items()})
        row['onset_chance'] = float(r['target'].mean())
        if r['spread'] is not None and 0 < r['spread'].sum() < len(scores):
            spread = logits[:, r['nbase']:r['nbase']+10].mean(1)
            row.update({f'spread10_{k}': v for k, v in retrieval(spread, r['spread']).items()})
        if include_anatomy and r['tissue'] is not None:
            tissue = r['tissue']
            top = expected_topk(scores, int(r['target'].sum()))
            row['WM_at_clinical_N'] = float(top[tissue == 0].sum() / top.sum())
            gm = r['target'] * (tissue == 1)
            if 0 < gm.sum() < len(scores):
                row.update({f'GM_onset_{k}': v for k, v in retrieval(scores, gm).items()})
                # Fixed GM target set and N; only known-WM candidates are demoted.
                oracle = scores.copy()
                oracle[tissue == 0] = scores.min() - 1
                row['GM_oracle_AccAtN'] = retrieval(oracle, gm)['AccAtN']
        rows.append(row)
        predictions.append({'patient': r['patient'], 'recording': r['recording'], 'onset': scores, 'logits': logits})
    return rows, predictions


def selection_metric(rows: list[dict], selection: str) -> float:
    metrics = ['onset_AP'] if selection == 'onset' else ['onset_AP', 'spread10_AP']
    frame = pd.DataFrame(rows)
    if any(m not in frame or frame[m].isna().any() for m in metrics):
        raise ValueError('Selection endpoint unavailable for some validation records')
    value = float(frame.groupby('patient')[metrics].mean().mean().mean())
    if not np.isfinite(value):
        raise ValueError('Nonfinite validation metric')
    return value


def train(model: torch.nn.Module, training: list[dict], validation: list[dict], config) -> tuple[dict, list[dict], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    groups = {}
    for record in training:
        groups.setdefault(record['patient'], []).append(record)
    patients = sorted(groups)
    if config.spread_weight and any(r['y_spread'] is None for r in training):
        raise ValueError('Spread supervision requires complete ten-second training targets')
    best, best_epoch, state, history = -np.inf, -1, None, []
    for epoch in range(config.epochs + 1):
        losses = []
        if epoch:
            model.train()
            if config.sampling == 'patient_uniform':
                # Sample each record immediately before its update, preserving the
                # legacy numpy RNG order while torch dropout uses its own RNG.
                order = np.random.permutation(len(patients))
                batches = (groups[patients[i]] for i in order)
                selected = (runs[np.random.randint(len(runs))] for runs in batches)
            else:
                selected = (training[i] for i in np.random.permutation(len(training)))
            for r in selected:
                logits = model(r['x'], r['nbase'])['localization']
                loss = onset_objective(logits, r['nbase'], r['y'], config.focal_gamma, config.rank_weight)
                if config.spread_weight:
                    loss = loss + config.spread_weight * balanced_binary(logits[:, r['nbase']:r['nbase']+10].mean(1), r['y_spread'], config.focal_gamma)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        metric = selection_metric(evaluate(model, validation)[0], config.selection)
        history.append({'epoch': epoch, 'validation_metric': metric, 'loss': float(np.mean(losses)) if losses else 0.})
        if metric > best + 1e-7:
            best, best_epoch, state = metric, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config.patience:
            break
    if state is None:
        raise ValueError('No valid validation-selected checkpoint')
    model.load_state_dict(state)
    return state, history, best_epoch
