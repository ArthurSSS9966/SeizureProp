"""Inference depends only on signal features and a checkpoint."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ..features.representations import FEATURE_NAMES, SCHEMA_ID, centered_features
from ..models.registry import build_model
from ..objectives.primitives import bag_logits
from ..schemas import validate_arrays


def predict_arrays(saved: dict, absolute: np.ndarray, relative: np.ndarray, nbase: int) -> dict[str, np.ndarray]:
    validate_arrays(absolute, relative, nbase)
    if 'feature_schema' in saved and saved['feature_schema'] != {'id': SCHEMA_ID, 'names': FEATURE_NAMES}:
        raise ValueError('Checkpoint representation mismatch')
    values = centered_features(absolute, relative, nbase)
    mean, scale = saved['mean'].numpy(), saved['scale'].numpy()
    if mean.shape != (values.shape[-1],) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid checkpoint scaler')
    normalized = np.clip((values - mean) / scale, -12, 12)
    model = build_model(saved.get('architecture', 'combined'), values.shape[-1], saved['hidden']).eval()
    model.load_state_dict(saved['model_state'])
    with torch.no_grad():
        logits = model(torch.tensor(normalized.transpose(1, 2, 0)), nbase)['localization']
        return {'localization_onset': logits[:, nbase].numpy(),
                'localization_involved': bag_logits(logits[:, nbase:]).numpy(),
                'localization_probability': logits.sigmoid().numpy()}


def predict(checkpoint: Path, features: Path, output: Path) -> dict[str, np.ndarray]:
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with np.load(features, allow_pickle=False) as data:
        result = predict_arrays(saved, data['absolute'], data['relative'], int(data['nbase']))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result
