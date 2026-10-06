"""Small strict JSON configuration; paths resolve against the config file."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from .artifacts.io import read_json


@dataclass(frozen=True)
class ExperimentConfig:
    manifest: str
    splits: str
    output: str
    protocol: str
    model: str = 'combined'
    hidden: int = 32
    seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 80
    patience: int = 15
    learning_rate: float = .001
    weight_decay: float = .001
    focal_gamma: float = 2.0
    rank_weight: float = .2
    spread_weight: float = 0.0
    selection: str = 'onset'
    sampling: str = 'patient_uniform'
    initializations: dict | None = None
    scaler_checkpoint: str | None = None
    initialization_provenance: str | None = None
    device: str = 'cpu'
    threads: int = 2

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        values = read_json(path)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        for name in ('manifest', 'splits', 'output', 'scaler_checkpoint', 'initialization_provenance'):
            if values.get(name):
                values[name] = str((path.parent / values[name]).resolve())
        if values.get('initializations'):
            values['initializations'] = {str(k): str((path.parent / v).resolve()) for k, v in values['initializations'].items()}
        values['seeds'] = tuple(values.get('seeds', (17, 29, 43)))
        config = cls(**values)
        if config.model not in {'combined', 'context'} or config.selection not in {'onset', 'joint'}:
            raise ValueError('Unknown model or selection endpoint')
        if config.sampling not in {'patient_uniform', 'recordings'}:
            raise ValueError('Unknown sampling protocol')
        if config.epochs < 0 or config.patience < 1 or config.learning_rate <= 0 or config.hidden < 1 or config.threads < 1:
            raise ValueError('Invalid training parameters')
        if not config.protocol or not config.seeds or len(set(config.seeds)) != len(config.seeds):
            raise ValueError('Protocol and distinct seeds required')
        if min(config.spread_weight, config.rank_weight, config.focal_gamma, config.weight_decay) < 0:
            raise ValueError('Weights cannot be negative')
        if config.initializations and set(config.initializations) != {str(s) for s in config.seeds}:
            raise ValueError('Provide one initialization per seed')
        if (config.initializations or config.scaler_checkpoint) and not config.initialization_provenance:
            raise ValueError('External weights/scaling require patient and checkpoint provenance')
        return config

    def as_dict(self) -> dict:
        return asdict(self)
