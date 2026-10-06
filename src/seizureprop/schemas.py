"""Validate representation and patient separation before fitting or inference."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .features.representations import FEATURE_NAMES, SCHEMA_ID


@dataclass(frozen=True)
class FeatureSpec:
    schema_id: str
    names: tuple[str, ...]
    units: str
    window_seconds: float = 1.0

    def validate(self) -> None:
        if self.schema_id != SCHEMA_ID or self.names != tuple(FEATURE_NAMES):
            raise ValueError('Unsupported or reordered feature schema')
        if self.units not in {'V', 'uV'} or self.window_seconds != 1.0:
            raise ValueError('Explicit supported amplitude units and one-second windows required')


def validate_arrays(absolute: np.ndarray, relative: np.ndarray, nbase: int) -> None:
    if absolute.ndim != 3 or absolute.shape != relative.shape or absolute.shape[-1] != len(FEATURE_NAMES):
        raise ValueError('Expected matching (seconds, channels, 17) signal features')
    if absolute.shape[1] < 2 or not 0 < nbase < len(absolute):
        raise ValueError('At least two channels and nonempty baseline/ictal phases required')
    if not np.isfinite(absolute).all() or not np.isfinite(relative).all():
        raise ValueError('Nonfinite features')


def validate_splits(splits: dict, patients: set[str]) -> None:
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Specify train, validation and test patients')
    groups = [set(splits[k]) for k in ('train', 'validation', 'test')]
    if any(not g or not g <= patients for g in groups):
        raise ValueError('Empty partition or patient missing from manifest')
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('Patient overlap between partitions')
    if set.union(*groups) != patients:
        raise ValueError('Manifest patients must be explicitly assigned to a partition')
