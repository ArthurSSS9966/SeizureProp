"""Reserved-baseline signal transforms for matched onset quality ablations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    sampling_rate: int = 256
    baseline_start_seconds: int = 2
    baseline_end_seconds: int = 22
    derivative_mad_threshold: float = 12.0
    amplitude_mad_threshold: float = 8.0
    hold_samples: int = 3
    scale_floor_uv: float = 1.0
    clip_limit: float = 20.0


def suppress_impulses(signal: np.ndarray, config: QualityConfig = QualityConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Hold the preceding value briefly at extreme, abrupt signal excursions.

    Input is time by channel in microvolts, after causal resampling and before
    bandpass filtering. Thresholds use only the reserved baseline. Decisions
    after calibration depend on current/past samples, never future samples.
    This is an experimental transient suppressor, not a validated artifact label:
    sharp physiological events may also be affected. No channel is removed.
    """
    values = np.asarray(signal, dtype=np.float64)
    start = config.baseline_start_seconds * config.sampling_rate
    stop = config.baseline_end_seconds * config.sampling_rate
    if values.ndim != 2 or len(values) < stop or not np.isfinite(values).all():
        raise ValueError('Finite time-by-channel signal and complete baseline required')
    baseline = values[start:stop]
    center = np.median(baseline, axis=0)
    amplitude = np.maximum(1.4826*np.median(np.abs(baseline-center), axis=0), config.scale_floor_uv)
    differences = np.diff(values, axis=0, prepend=values[:1])
    ref = differences[start:stop]
    dcenter = np.median(ref, axis=0)
    dscale = np.maximum(1.4826*np.median(np.abs(ref-dcenter), axis=0), config.scale_floor_uv)
    trigger = ((np.abs(differences-dcenter) > config.derivative_mad_threshold*dscale)
               & (np.abs(values-center) > config.amplitude_mad_threshold*amplitude))
    mask = trigger.copy()
    for lag in range(1, config.hold_samples):
        mask[lag:] |= trigger[:-lag]
    mask[0] = False
    # Index the most recent unmasked sample; no interpolation from future data.
    indices = np.broadcast_to(np.arange(len(values))[:,None], values.shape).copy()
    indices[mask] = 0
    indices = np.maximum.accumulate(indices, axis=0)
    clean = np.take_along_axis(values, indices, axis=0)
    return clean, mask


def normalize_baseline(signal: np.ndarray, robust: bool = False,
                       config: QualityConfig = QualityConfig()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Scale a filtered signal using its reserved baseline, without label access."""
    x = np.asarray(signal, dtype=np.float64)
    baseline = x[config.baseline_start_seconds*config.sampling_rate:
                 config.baseline_end_seconds*config.sampling_rate]
    if x.ndim != 2 or len(baseline) != (config.baseline_end_seconds-config.baseline_start_seconds)*config.sampling_rate:
        raise ValueError('Complete time-by-channel baseline required')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite waveform')
    center = np.median(baseline, axis=0) if robust else baseline.mean(0)
    scale = 1.4826*np.median(np.abs(baseline-center), axis=0) if robust else baseline.std(0)
    scale = np.maximum(scale, config.scale_floor_uv)
    standardized = (x-center)/scale
    return np.clip(standardized, -config.clip_limit, config.clip_limit).astype(np.float32), {
        'center': center, 'scale': scale,
        'clipped_fraction': np.mean(np.abs(standardized)>=config.clip_limit, axis=0)}
