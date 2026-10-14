"""Unit-explicit, signal-only features for a separate tissue classification stage."""
from __future__ import annotations

import numpy as np

TISSUE_SCHEMA = "tissue_median_context_iqr_uv_v1"


def to_microvolts(absolute: np.ndarray, units: str) -> np.ndarray:
    """Convert the 17-feature representation without silently converting floors.

    Amplitude logs shift by ln(scale), power logs by twice that amount. Ratios,
    entropy and zero-crossing rate do not change. Source floor-clipped values are
    not invertible; converting those requires rebuilding from the raw waveform.
    """
    values = np.asarray(absolute, dtype=np.float64).copy()
    if values.ndim != 3 or values.shape[-1] != 17 or not np.isfinite(values).all():
        raise ValueError("Expected finite time-by-channel-by-17 feature array")
    if units not in {"V", "uV"}:
        raise ValueError("Explicit V or uV units required")
    if units == "V":
        if (values[..., :3] <= np.log(1e-10) + 1e-5).any() or (values[..., 4:10] <= np.log(1e-20) + 1e-5).any():
            raise ValueError("Floor-clipped source features require raw-waveform re-extraction")
        values[..., :3] += np.log(1e6)
        values[..., 4:10] += 2 * np.log(1e6)
    return values


def aggregate_tissue(segment: np.ndarray) -> np.ndarray:
    """One row per bipolar channel; no tissue or seizure labels are inputs."""
    values = np.asarray(segment, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 17 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite feature frames required")
    median = np.median(values, axis=0)
    context = median - np.median(median, axis=0, keepdims=True)
    iqr = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    return np.concatenate([median, context, iqr], axis=-1)
