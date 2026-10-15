"""Patient-level cohort design and reproducible, auditable training mixtures."""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _validate(patients: pd.DataFrame, stratum: str) -> None:
    if patients.empty or patients.patient.isna().any() or patients.patient.duplicated().any():
        raise ValueError("Exactly one row per nonmissing patient is required")
    if patients[stratum].isna().any():
        raise ValueError("Missing strata must be explicitly represented as unknown")


def assert_patient_isolation(partitions: Mapping[str, Sequence[str]]) -> None:
    """Reject patient leakage, including between pretraining and held-out roles."""
    seen: set[str] = set()
    for name, values in partitions.items():
        current = set(values)
        if not current or len(current) != len(values):
            raise ValueError(f"Empty or duplicated patient membership in {name}")
        if seen & current:
            raise ValueError(f"Patient overlap in {name}: {sorted(seen & current)}")
        seen.update(current)


def _metadata_matrix(patients: pd.DataFrame, stratum: str,
                     numeric: Sequence[str]) -> np.ndarray:
    """Represent metadata for split design only, never for signal normalization."""
    categorical = pd.get_dummies(patients[stratum], dtype=float).to_numpy()
    values = patients[list(numeric)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite metadata is invalid")
    missing = np.isnan(values)
    # Neutral fill and a separate missingness indicator retain unknown categories.
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                       for col in values.T])
    values = np.where(missing, median, values)
    values = (values-values.mean(0)) / np.maximum(values.std(0), 1e-8)
    return np.column_stack((categorical, values, missing.astype(float)))


def stratified_patient_folds(patients: pd.DataFrame, stratum: str,
                             numeric: Sequence[str] = (), n_splits: int = 3,
                             seed: int = 47, candidates: int = 256) -> dict[str, int]:
    """Keep subtype counts stratified and balance secondary metadata.

    Candidate splits are selected using metadata alone, never signal features,
    target scores, or model performance. Rare primary strata are refused rather
    than silently split across patients or merged into a different diagnosis.
    """
    _validate(patients, stratum)
    patients = patients.sort_values('patient').reset_index(drop=True)
    if n_splits < 2 or candidates < 1 or patients[stratum].value_counts().min() < n_splits:
        raise ValueError("Each primary stratum needs at least n_splits patients")
    matrix = _metadata_matrix(patients, stratum, numeric)
    best_score, best = float('inf'), None
    for candidate in range(candidates):
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed+candidate)
        assignment = np.zeros(len(patients), dtype=int)
        score = 0.
        for fold, (_, held) in enumerate(folds.split(matrix, patients[stratum])):
            assignment[held] = fold
            score += float(np.square(matrix[held].mean(0)-matrix.mean(0)).mean())
        if score < best_score:
            best_score, best = score, assignment.copy()
    return dict(zip(patients.patient, best.tolist()))


def stratified_validation_patients(patients: pd.DataFrame, stratum: str,
                                  numeric: Sequence[str] = (), size: int = 3,
                                  seed: int = 47) -> list[str]:
    """Choose an inner validation set with each observed stratum on both sides.

    Exhaustive selection is intended for small clinical cohorts. The explicit
    size bound prevents accidental combinatorial work on foundation-scale data.
    """
    _validate(patients, stratum)
    patients = patients.sort_values('patient').reset_index(drop=True)
    if len(patients) > 30 or not 0 < size < len(patients):
        raise ValueError("Use a scalable splitter above 30 patients; invalid holdout size")
    matrix = _metadata_matrix(patients, stratum, numeric)
    strata = set(patients[stratum])
    rng = np.random.default_rng(seed)
    best, best_score = None, (float('inf'), float('inf'))
    proportions = patients[stratum].value_counts(normalize=True)
    for combo in combinations(range(len(patients)), size):
        held = np.array(combo)
        fit = np.setdiff1d(np.arange(len(patients)), held)
        if set(patients.iloc[held][stratum]) != strata or set(patients.iloc[fit][stratum]) != strata:
            continue
        held_proportions = patients.iloc[held][stratum].value_counts(normalize=True)
        primary = float(((held_proportions-proportions)**2).sum())
        secondary = float(np.square(matrix[held].mean(0)-matrix.mean(0)).mean())
        score = (primary, secondary + float(rng.uniform(0, 1e-10)))
        if score < best_score:
            best_score, best = score, patients.iloc[held].patient.tolist()
    if best is None:
        raise ValueError("Insufficient patient coverage for stratified validation")
    return best


def patient_probabilities(patient_groups: Mapping[str, str], policy: str) -> dict[str, float]:
    """Return intended contribution: equal patients or equal subtype groups."""
    if not patient_groups or any(pd.isna(g) for g in patient_groups.values()):
        raise ValueError("Nonempty patients with explicit groups required")
    if policy == 'patient':
        return {p: 1/len(patient_groups) for p in sorted(patient_groups)}
    if policy != 'subtype':
        raise ValueError("Unknown sampling policy")
    counts = pd.Series(list(patient_groups.values())).value_counts()
    return {p: 1/(len(counts)*counts[g]) for p, g in sorted(patient_groups.items())}


def patient_epoch_schedule(patient_groups: Mapping[str, str], policy: str,
                           draws: int, seed: int) -> list[str]:
    """Use balanced integer quotas, include every patient, and randomize order.

    All events of a selected patient can be processed with equal within-patient
    weight. Both policies must receive the same draw budget. Finite-epoch quota
    remainders are assigned randomly with a reproducible seed; realized counts
    should be logged alongside intended probabilities.
    """
    patient_probabilities(patient_groups, policy)
    if draws < len(patient_groups):
        raise ValueError("Budget cannot cover every patient")
    rng = np.random.default_rng(seed)
    groups = ({'all': sorted(patient_groups)} if policy == 'patient' else
              {g: sorted(p for p, v in patient_groups.items() if v == g)
               for g in sorted(set(patient_groups.values()))})
    quotas = dict.fromkeys(groups, draws // len(groups))
    for g in rng.permutation(list(groups))[:draws % len(groups)]:
        quotas[g] += 1
    result: list[str] = []
    for group, patients in groups.items():
        quota = quotas[group]
        if quota < len(patients):
            raise ValueError("Increase budget to cover every patient under equal group quotas")
        result.extend(patients * (quota // len(patients)))
        result.extend(rng.permutation(patients)[:quota % len(patients)].tolist())
    return rng.permutation(result).tolist()


def coverage_table(records: pd.DataFrame, dimensions: Sequence[str]) -> pd.DataFrame:
    """Count patients and events by role and metadata level, including unknowns.

    A patient with multiple acquisition rates can appear in several rate rows;
    these marginal counts must not be added as independent patient totals.
    """
    rows = []
    for role, subset in records.groupby('role'):
        for dimension in dimensions:
            values = subset[dimension].fillna('unknown').astype(str)
            for level, group in subset.groupby(values):
                rows.append({'role': role, 'dimension': dimension, 'level': level,
                             'patients': group.patient.nunique(), 'events': group.recording.nunique(),
                             'patient_fraction': group.patient.nunique()/subset.patient.nunique()})
    return pd.DataFrame(rows)
