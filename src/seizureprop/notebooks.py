"""Explicit setup for notebooks relocated below the repository root."""
from __future__ import annotations
import os
from pathlib import Path
import sys
from .legacy import enable_legacy_imports


def initialize(root: Path | None = None) -> Path:
    """Restore historical relative paths only when a notebook requests setup."""
    if root is None:
        here = Path.cwd().resolve()
        candidates = [here, *here.parents, Path(__file__).resolve().parents[2]]
        root = next((p for p in candidates if (p / 'pyproject.toml').is_file()
                     and (p / 'src/seizureprop').is_dir()), None)
    if root is None:
        raise FileNotFoundError('Run inside the checkout or pass initialize(Path("..."))')
    root = Path(root).resolve()
    if not (root / 'src/seizureprop').is_dir():
        raise ValueError(f'Not a SeizureProp checkout: {root}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    enable_legacy_imports()
    os.chdir(root)
    return root
