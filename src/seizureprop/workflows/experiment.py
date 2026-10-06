"""Compose validated inputs, training and held-out evaluation in immutable runs."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from ..artifacts.io import (read_json, write_json, file_hash, fingerprint, code_hashes,
                            environment, completed_run, mark_complete)
from ..config import ExperimentConfig
from ..data.manifest import load_records
from ..features.representations import FEATURE_NAMES, SCHEMA_ID
from ..models.registry import build_model, initialize
from ..schemas import validate_splits
from ..training.localization import fit_scaler, tensor_records, train, evaluate


def initialization_lineage(config: ExperimentConfig, splits: dict) -> dict:
    paths = set((config.initializations or {}).values())
    if config.scaler_checkpoint:
        paths.add(config.scaler_checkpoint)
    if not paths:
        return {}
    provenance = read_json(Path(config.initialization_provenance))
    # BIDS prefix is a formatting alias, not a different patient.
    canonical = lambda p: str(p).removeprefix('sub-')
    held_out = {canonical(p) for role in ('validation', 'test') for p in splits[role]}
    learned = {canonical(p) for p in provenance['fit_patients'] + provenance['selection_patients']}
    if held_out & learned:
        raise ValueError(f'External fitting/selection overlaps held-out patients: {held_out & learned}')
    hashes = {}
    for path in sorted(paths):
        digest = file_hash(Path(path))
        if provenance['checkpoints'].get(str(Path(path).resolve())) != digest:
            raise ValueError('Initialization is not pinned by its provenance')
        hashes[path] = digest
    return {'checkpoints': hashes, 'provenance_sha256': file_hash(Path(config.initialization_provenance)),
            'fit_patients': provenance['fit_patients'], 'selection_patients': provenance['selection_patients']}


def run(config_path: Path) -> dict:
    config = ExperimentConfig.load(config_path.resolve())
    splits = read_json(Path(config.splits))
    manifest = read_json(Path(config.manifest))
    validate_splits(splits, {r['patient'] for r in manifest['records']})
    lineage = initialization_lineage(config, splits)
    records, manifest = load_records(Path(config.manifest))
    groups = {k: [r for r in records if r['patient'] in patients] for k, patients in splits.items()}
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if config.device != 'cpu':
        raise ValueError('Version 0.1 runner supports reproducible CPU experiments only')
    if config.scaler_checkpoint:
        scaler = torch.load(config.scaler_checkpoint, map_location='cpu', weights_only=True)
        mean, scale = scaler['mean'].numpy(), scaler['scale'].numpy()
    else:
        mean, scale = fit_scaler(groups['train'])
    size = records[0]['features'].shape[-1]
    if mean.shape != (size,) or scale.shape != (size,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid training scaler')
    parts = {k: tensor_records(v, mean, scale, config.device) for k, v in groups.items()}
    env = environment()
    identity = {'config': config.as_dict(), 'manifest_sha256': file_hash(Path(config.manifest)),
                'splits_sha256': file_hash(Path(config.splits)), 'code': code_hashes(),
                'initialization': lineage, 'environment': env}
    signature = fingerprint(identity)
    output = Path(config.output)
    if completed_run(output, signature):
        return read_json(output / 'summary.json')
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', {'signature': signature, **identity})
    write_json(output / 'resolved_config.json', config.as_dict())
    write_json(output / 'splits.json', splits)
    all_rows = []
    for seed in config.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(config.model, size, config.hidden).to(config.device)
        if config.initializations:
            initial = torch.load(config.initializations[str(seed)], map_location='cpu', weights_only=True)
            initialize(model, initial['model_state'])
        started = time.monotonic()
        state, history, epoch = train(model, parts['train'], parts['validation'], config)
        directory = output / f'seed{seed}'
        directory.mkdir()
        torch.save({'artifact_version': 1, 'model_state': state, 'hidden': config.hidden,
                    'architecture': config.model, 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                    'feature_schema': {'id': SCHEMA_ID, 'names': FEATURE_NAMES}, 'signal_units': manifest['features']['units'],
                    'config': config.as_dict(), 'seed': seed, 'signature': signature}, directory / 'best.pt')
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        pd.DataFrame(evaluate(model, parts['validation'])[0]).to_csv(directory / 'validation_metrics.csv', index=False)
        rows, predictions = evaluate(model, parts['test'], include_anatomy=True)
        pd.DataFrame(rows).to_csv(directory / 'test_metrics.csv', index=False)
        prediction_dir = directory / 'predictions'
        prediction_dir.mkdir()
        for index, prediction in enumerate(predictions):
            np.savez_compressed(prediction_dir / f'{index:04d}.npz', **prediction)
        all_rows.extend({**r, 'seed': seed} for r in rows)
        write_json(directory / 'fit.json', {'best_epoch': epoch, 'seconds': time.monotonic()-started})
        print(f'{output.name} seed {seed}: epoch {epoch}, {time.monotonic()-started:.1f}s', flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / 'metrics.csv', index=False)
    metrics = [c for c in frame if c not in {'patient', 'recording', 'seed'}]
    # First average seeds/recordings within a patient; each patient gets one vote.
    patients = frame.groupby('patient')[metrics].mean()
    patients.to_csv(output / 'patient_metrics.csv')
    summary = {'protocol': config.protocol, 'test_patients': len(patients),
               'test_recordings': len(groups['test']), 'seeds': list(config.seeds),
               'metrics': {k: float(v) for k, v in patients.mean().items()},
               'endpoint_patient_counts': {k: int(v) for k, v in patients.count().items()},
               'endpoint_recording_seed_counts': {k: int(v) for k, v in frame[metrics].count().items()}}
    write_json(output / 'summary.json', summary)
    mark_complete(output)
    return summary
