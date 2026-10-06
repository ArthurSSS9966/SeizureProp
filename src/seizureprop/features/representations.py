"""Versioned legacy-compatible signal representations."""
import numpy as np
AMPLITUDE_INDICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
FEATURE_NAMES = ["log_rms", "log_line_length", "log_ptp", "zero_crossing", "log_delta", "log_theta", "log_alpha", "log_beta", "log_gamma", "log_high_gamma", "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma", "rel_high_gamma", "spectral_entropy"]
SCHEMA_ID = "centered_absolute_relative_v1"

def extract_features(signal: np.ndarray, fs: int) -> np.ndarray:
    """Return (time, channels, features) for non-overlapping one-second frames."""
    n = len(signal) // fs
    x = signal[:n * fs].reshape(n, fs, -1).transpose(0, 2, 1)
    x = x - x.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(x * np.hanning(fs), axis=-1)) ** 2 / fs
    freq = np.fft.rfftfreq(fs, 1 / fs)
    # Exclude line-noise bins consistently from spectral estimates.
    keep = (freq >= 1) & (freq <= 120) & (np.abs(freq - 60) > 2) & (np.abs(freq - 120) > 2)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 70), (70, 120)]
    bp = np.stack([power[..., keep & (freq >= lo) & (freq < hi)].sum(-1)
                   for lo, hi in bands], -1)
    relative = bp / np.maximum(bp.sum(-1, keepdims=True), 1e-20)
    p = power[..., keep]
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-20)
    entropy = -(p * np.log(np.maximum(p, 1e-20))).sum(-1) / np.log(p.shape[-1])
    values = [np.log(np.maximum(np.sqrt((x ** 2).mean(-1)), 1e-10)),
              np.log(np.maximum(np.abs(np.diff(x, axis=-1)).mean(-1), 1e-10)),
              np.log(np.maximum(np.ptp(x, axis=-1), 1e-10)),
              (np.diff(np.signbit(x), axis=-1) != 0).mean(-1)]
    features = np.concatenate([np.stack(values, -1), np.log(np.maximum(bp, 1e-20)),
                               relative, entropy[..., None]], -1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite signal features")
    return features

def baseline_relative(features: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Fit channel-specific location/scale exclusively on reserved baseline."""
    center = np.median(calibration, axis=0)
    scale = np.quantile(calibration, .75, axis=0) - np.quantile(calibration, .25, axis=0)
    # Floors guard nearly constant baseline features without consulting ictal data.
    floor = np.array([.1, .1, .1, .02] + [.2] * 6 + [.02] * 6 + [.02])
    return np.clip((features - center) / np.maximum(scale, floor), -12, 12).astype(np.float32)

def signal_features(absolute: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """Retain baseline tissue cues alongside changes relative to baseline."""
    if absolute.ndim != 3 or absolute.shape != relative.shape:
        raise ValueError("Absolute/relative features must have matching (time, pairs, features) shape")
    values = np.concatenate([absolute, relative], axis=-1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite signal features")
    return values

def centered_features(absolute: np.ndarray, relative: np.ndarray, nbase: int) -> np.ndarray:
    """Remove recording-wide amplitude offsets using unlabeled baseline only.

    Preserve relative amplitude differences between pairs. No channel identities,
    anatomy, clinical labels or future ictal values enter this centering step.
    """
    if not 0 < nbase < len(absolute):
        raise ValueError('Nonempty baseline and ictal intervals required')
    values = signal_features(absolute, relative)
    center = np.median(absolute[:nbase, :, AMPLITUDE_INDICES], axis=(0, 1))
    values[:, :, AMPLITUDE_INDICES] -= center
    return values
