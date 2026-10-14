"""Export one reusable tissue label per bipolar channel from signal features."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ..features.tissue import TISSUE_SCHEMA, aggregate_tissue, to_microvolts


def predict_tissue_arrays(saved: dict, absolute: np.ndarray, names: np.ndarray,
                         units: str, segment_kind: str, nbase: int | None = None) -> pd.DataFrame:
    """Use a baseline segment once; anatomy and clinical targets are never read.

    Binary checkpoints cannot identify mixed anatomy. Three-class checkpoints
    output an argmax composition label separately from the GM retention rule.
    UNCERTAIN in binary output is a score range, not an atlas-derived class.
    Ictal inference uses the first six seconds and is explicitly retrospective.
    """
    if saved.get("schema") != TISSUE_SCHEMA or saved.get("units") != "uV":
        raise ValueError("Expected a unit-corrected tissue checkpoint")
    state = saved["state"]
    if state not in {"baseline30", "baseline6", "ictal6"}:
        raise ValueError("Unsupported tissue model state")
    if segment_kind not in {"baseline", "ictal"} or (state.startswith("baseline") != (segment_kind == "baseline")):
        raise ValueError("Segment kind must match the checkpoint training state")
    values = np.asarray(absolute)
    if values.ndim != 3 or values.shape[-1] != 17:
        raise ValueError("Expected time-by-channel-by-17 features")
    if len(names) != values.shape[1] or len(set(map(str, names))) != len(names):
        raise ValueError("Unique channel names must match signal columns")
    if nbase is not None and not 0 < nbase < len(values):
        raise ValueError("Invalid baseline boundary")
    if segment_kind == "baseline":
        values = values if nbase is None else values[:nbase]
        seconds = 30 if state == "baseline30" else 6
    else:
        if nbase is None:
            raise ValueError("Ictal inference needs an explicit baseline boundary")
        values = values[nbase:nbase + 6]
        seconds = 6
    if len(values) < seconds:
        raise ValueError("Insufficient signal for the trained tissue window")
    values = to_microvolts(values, units)
    segments = [aggregate_tissue(values[start:start + seconds])
                for start in range(0, len(values) - seconds + 1, seconds)]
    features = np.median(np.stack(segments), axis=0)
    threshold = float(saved["threshold"])
    if tuple(getattr(saved["model"], "allowed_labels", (0, 1))) == (0, 1, 2):
        if tuple(saved.get("classes", ())) != ("WM", "GM", "MIXED"):
            raise ValueError("Three-class checkpoint must declare WM, GM, MIXED order")
        probabilities = saved["model"].predict_proba(features)
        scores = probabilities[:, 1]
        return pd.DataFrame({"pair": names, "gm_score": scores,
                             "wm_score": probabilities[:, 0], "mixed_score": probabilities[:, 2],
                             "channel_label": np.array(saved["classes"])[probabilities.argmax(axis=1)],
                             "retain": scores >= threshold, "threshold": threshold,
                             "segments": len(segments), "units": "uV", "model_state": state})
    scores = saved["model"].predict(features)
    labels = np.where(scores < threshold, "WM", np.where(scores >= max(.5, threshold), "GM", "UNCERTAIN"))
    return pd.DataFrame({"pair": names, "gm_score": scores, "channel_label": labels,
                         "retain": scores >= threshold, "threshold": threshold,
                         "segments": len(segments), "units": "uV", "model_state": state})


def predict_tissue(checkpoint: Path, features: Path, output: Path, units: str,
                   segment_kind: str) -> dict:
    """Load a trusted project checkpoint and export a private channel-label CSV."""
    if output.exists():
        raise ValueError("Preserve existing labels; choose a new output path")
    saved = joblib.load(checkpoint)
    with np.load(features, allow_pickle=False) as data:
        if "units" in data and str(data["units"]) != units:
            raise ValueError("Declared units conflict with the feature file")
        frame = predict_tissue_arrays(saved, data["absolute"], data["names"], units,
                                     segment_kind, int(data["nbase"]) if "nbase" in data else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {"channels": len(frame), "labels": frame.channel_label.value_counts().to_dict(),
            "output": str(output), "input": "Signal features only; labels are per bipolar pair"}
