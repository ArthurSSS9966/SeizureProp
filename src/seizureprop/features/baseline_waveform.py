"""Signal-only baseline representations for contact and bipolar tissue probes."""
from __future__ import annotations

from fractions import Fraction
import itertools

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, welch
from scipy.stats import kurtosis

from .representations import extract_features
from .tissue import aggregate_tissue

SCHEMA = "baseline_waveform_uv_256_v1"
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 55), (65, 90), (90, 115))


def prepare_baseline(signal: np.ndarray, fs: float, units: str) -> np.ndarray:
    """Resample a baseline-only segment to 256 Hz, bandpass, discard warmup.

    The caller must slice out all ictal samples BEFORE this function: polyphase
    resampling is noncausal. A segment has two warmup seconds plus 30 seconds
    retained. Input shape is samples by channels. No fitted normalization.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or fs < 256 or units not in {"V", "uV"}:
        raise ValueError("Finite sample-by-channel baseline, fs>=256 and explicit units required")
    if len(x) < round(32 * fs):
        raise ValueError("Need 32 baseline seconds including warmup")
    x = x[:round(32 * fs)] * (1e6 if units == "V" else 1.)
    ratio = Fraction(256 / fs).limit_denominator(10000)
    x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
    x = sosfilt(butter(4, [1, 120], fs=256, btype="bandpass", output="sos"), x, axis=0)
    return x[512:8192]


def rich_features(signal: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Spectral shape, temporal variability, Hjorth and permutation features.

    Log-PSD slopes are descriptive regressions, not fitted aperiodic components.
    Context features center each descriptor across the current unlabeled montage.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or len(x) != 7680 or not np.isfinite(x).all():
        raise ValueError("Expected 30 seconds at 256 Hz, samples by channels")
    basic = aggregate_tissue(extract_features(x, 256))
    freq, power = welch(x, fs=256, nperseg=512, noverlap=256, axis=0)
    keep = (freq >= 1) & (freq <= 115) & (np.abs(freq - 60) > 3)
    psd = np.maximum(power[keep], 1e-20)
    f = freq[keep]
    total = psd.sum(0)
    relative = psd / total
    extra, names = [], []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        extra.append(values)

    for lo, hi in BANDS:
        bp = psd[(f >= lo) & (f < hi)].sum(0)
        add(f"welch_log_{lo}_{hi}", np.log(np.maximum(bp, 1e-20)))
        add(f"welch_relative_{lo}_{hi}", bp / total)
    for lo, hi in [(2, 45), (65, 110)]:
        mask = (f >= lo) & (f <= hi)
        a = np.log(f[mask]); a -= a.mean()
        slope = a @ np.log(psd[mask]) / (a @ a)
        add(f"log_psd_slope_{lo}_{hi}", slope)
    cdf = relative.cumsum(0)
    for q in (.5, .9, .95):
        add(f"spectral_edge_{q}", f[(cdf >= q).argmax(0)] / 128.)
    add("spectral_flatness", np.exp(np.log(psd).mean(0)) / psd.mean(0))
    add("welch_entropy", -(relative * np.log(relative)).sum(0) / np.log(len(f)))
    x = x - np.median(x, axis=0)
    var = np.maximum(np.var(x, axis=0), 1e-20)
    d = np.diff(x, axis=0)
    dvar = np.maximum(np.var(d, axis=0), 1e-20)
    mobility = np.sqrt(dvar / var)
    add("hjorth_mobility", mobility)
    add("hjorth_complexity", np.sqrt(np.var(np.diff(d, axis=0), axis=0) / dvar) / np.maximum(mobility, 1e-10))
    add("kurtosis", np.nan_to_num(kurtosis(x, axis=0), nan=0., posinf=0., neginf=0.))
    mad = np.maximum(np.median(np.abs(x), axis=0), 1e-10)
    add("tail_q99_mad", np.quantile(np.abs(x), .99, axis=0) / mad)
    add("burst_fraction", (np.abs(x) > 5 * mad).mean(0))
    for lag in (1, 4, 16, 64):
        add(f"autocorr_{lag}", np.mean(x[lag:] * x[:-lag], axis=0) / var)
    # Six ordinal patterns of length three; downsample to 64 Hz after bandpass.
    # This is an ordinal descriptor, not a second waveform for spectral analysis.
    z = x[::4]
    order = np.argsort(np.stack([z[:-2], z[1:-1], z[2:]], axis=-1), axis=-1, kind="stable")
    counts = np.stack([(order == pattern).all(-1).mean(0)
                       for pattern in itertools.permutations(range(3))])
    add("permutation_entropy_3", -(counts * np.log(np.maximum(counts, 1e-20))).sum(0) / np.log(6))
    e = np.stack(extra, axis=1)
    result = np.concatenate([basic, e, e - np.median(e, axis=0)], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite baseline descriptor")
    return result, [f"legacy_{i}" for i in range(51)] + names + [f"context_{n}" for n in names]


def contact_features(signal: np.ndarray, neighbors: list[list[int]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Contact descriptors using a common median reference and local differences.

    Neighbor indices encode immediately adjacent recorded contacts on each shaft,
    not anatomical labels. Reference subtraction removes any common additive
    recording reference. A one-neighbor edge uses a first difference; two use
    the local second difference. Missing neighbors receive a validity indicator.
    """
    x = np.asarray(signal, dtype=float)
    if len(neighbors) != x.shape[1] or x.shape[1] < 2:
        raise ValueError("Neighbors must match at least two contacts")
    referenced = x - np.median(x, axis=1, keepdims=True)
    rich, names = rich_features(referenced)
    local = np.zeros_like(referenced)
    corr, valid = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for j, adjacent in enumerate(neighbors):
        if any(i < 0 or i >= x.shape[1] or i == j for i in adjacent) or len(set(adjacent)) != len(adjacent):
            raise ValueError("Invalid neighbor indices")
        if adjacent:
            other = referenced[:, adjacent].mean(1)
            local[:, j] = referenced[:, j] - other
            a, b = referenced[:, j] - referenced[:, j].mean(), other - other.mean()
            corr[j] = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20)
            valid[j] = len(adjacent)
    local_rich, local_names = rich_features(local)
    # Missing local contrasts have explicit zero descriptors and an indicator.
    local_rich[valid == 0] = 0
    result = np.concatenate([rich, local_rich, corr[:, None], valid[:, None]], axis=1)
    return rich[:, :51], result, names + ["local_" + n for n in local_names] + ["neighbor_correlation", "neighbor_count"]
