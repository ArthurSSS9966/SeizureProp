"""Validated, hash-pinned feature manifests. No fitting occurs while loading."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..artifacts.io import file_hash, read_json, write_json
from ..features.representations import FEATURE_NAMES, SCHEMA_ID, centered_features
from ..schemas import FeatureSpec, validate_arrays


def load_records(path: Path) -> tuple[list[dict], dict]:
    manifest = read_json(path)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version')
    spec = manifest['features']
    FeatureSpec(spec['schema_id'], tuple(spec['names']), spec['units'], spec['window_seconds']).validate()
    records, identities = [], set()
    for entry in manifest['records']:
        cache = (path.parent / entry['cache']).resolve()
        if file_hash(cache) != entry['sha256']:
            raise ValueError(f'Changed cache: {cache}')
        with np.load(cache, allow_pickle=False) as data:
            # Deliberate whitelist: annotations never enter the feature builder.
            absolute, relative = data['absolute'].copy(), data['relative'].copy()
            nbase = int(data['nbase'])
            names = data['names'].copy()
            target = data[entry['onset_key']].copy()
            spread = data[entry['spread_key']].copy() if entry.get('spread_key') else None
            tissue = data['tissue'].copy() if 'tissue' in data else None
        validate_arrays(absolute, relative, nbase)
        n = absolute.shape[1]
        if len(names) != n or len(set(map(str, names))) != n:
            raise ValueError('Missing or duplicate channel identities')
        for label in [target] + ([] if spread is None else [spread]):
            if label.shape != (n,) or not np.isin(label, [0, 1]).all():
                raise ValueError('This protocol requires binary channel targets')
        if not 0 < target.sum() < n:
            raise ValueError('Onset ranking needs positive and negative candidates')
        if tissue is not None and (tissue.shape != (n,) or not np.isin(tissue, [-1, 0, 1]).all()):
            raise ValueError('Invalid tissue evaluation annotations')
        identity = (entry['patient'], entry['recording'])
        if identity in identities:
            raise ValueError('Duplicate recording identity')
        identities.add(identity)
        if spread is not None and len(absolute) - nbase < 10:
            raise ValueError('Ten-second spread target exceeds available observation window')
        records.append({**entry, 'absolute': absolute, 'relative': relative, 'nbase': nbase,
                        'names': names, 'target': target, 'spread': spread, 'tissue': tissue,
                        'features': centered_features(absolute, relative, nbase)})
    if not records:
        raise ValueError('Empty feature manifest')
    return records, manifest


def prepare_legacy(source: Path, root: Path, output: Path, kind: str, units: str) -> dict:
    """Adopt audited legacy feature caches without silently changing their units.

    This stage validates existing prepared features, not raw EDF preprocessing.
    Short multi-expert recordings retain valid onset targets, but their unsupported
    ten-second target is explicitly omitted. Clinical targets are never GM-filtered.
    """
    if output.exists():
        raise ValueError('Preserve existing prepared manifest; choose a new path')
    if kind == 'multiexpert':
        import json
        rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
        rows = [r for r in rows if r['status'] == 'included']
    elif kind == 'hup':
        rows = read_json(source)['records']
    else:
        raise ValueError('Supported cache adapters: hup, multiexpert')
    records, notes = [], []
    for r in rows:
        cache = (root / r['cache']).resolve()
        if file_hash(cache) != r['cache_sha256']:
            raise ValueError(f'Legacy cache hash mismatch: {cache}')
        with np.load(cache, allow_pickle=False) as data:
            duration = len(data['absolute']) - int(data['nbase'])
        spread_key = 'y_spread10' if kind == 'multiexpert' and duration >= 10 else None
        if kind == 'multiexpert' and spread_key is None:
            notes.append({'recording': r['seizure_id'], 'reason': 'spread10 unavailable: fewer than 10 seconds', 'ictal_seconds': duration})
        records.append({'patient': str(r['patient']), 'recording': r.get('seizure_id', cache.stem),
                        'cache': str(cache), 'sha256': r['cache_sha256'],
                        'onset_key': 'y_onset' if kind == 'multiexpert' else 'y_soz',
                        'spread_key': spread_key, 'center': r.get('center', 'HUP'),
                        'stim_induced': r.get('stim_induced'), 'ictal_seconds': duration})
    manifest = {'schema_version': 1, 'adapter': kind, 'source': str(source.resolve()),
                'source_sha256': file_hash(source), 'features': {'schema_id': SCHEMA_ID,
                'names': FEATURE_NAMES, 'units': units, 'window_seconds': 1.0},
                'targets': 'Clinical/consensus onset across ALL eligible channels; anatomy is evaluation-only',
                'records': records, 'endpoint_exclusions': notes}
    write_json(output, manifest)
    load_records(output)
    return manifest
