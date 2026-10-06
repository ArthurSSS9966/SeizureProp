"""Portable, explicit artifact I/O and immutable run fingerprints."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Replace JSON atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))}


def environment() -> dict:
    import scipy
    import pandas
    import sklearn
    return {'python': platform.python_version(), 'executable': sys.executable,
            'numpy': np.__version__, 'torch': str(torch.__version__),
            'scipy': scipy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__}


def completed_run(directory: Path, signature: str) -> bool:
    """Refuse stale or partial results instead of silently treating them as done."""
    if not directory.exists():
        return False
    manifest = directory / 'manifest.json'
    if not manifest.exists() or read_json(manifest)['signature'] != signature:
        raise ValueError(f'Existing run has incompatible lineage: {directory}')
    marker = directory / 'complete.json'
    if not marker.exists():
        raise ValueError(f'Incomplete run; preserve it and use a fresh output: {directory}')
    for name, expected in read_json(marker)['outputs'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Completed output changed: {name}')
    return True


def mark_complete(directory: Path) -> None:
    outputs = {p.relative_to(directory).as_posix(): file_hash(p)
               for p in directory.rglob('*') if p.is_file() and p.name != 'complete.json'}
    write_json(directory / 'complete.json', {'outputs': outputs})
