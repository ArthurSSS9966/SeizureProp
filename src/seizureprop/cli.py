"""Command-line entry points with no import-time experiment execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproducible sEEG-only localization')
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('run', help='Train, validation-select, evaluate and record an experiment')
    training.add_argument('--config', type=Path, required=True)
    preparation = commands.add_parser('prepare', help='Validate and adopt pinned legacy feature caches')
    for name in ('source', 'root', 'output'):
        preparation.add_argument(f'--{name}', type=Path, required=True)
    preparation.add_argument('--kind', choices=['hup', 'multiexpert'], required=True)
    preparation.add_argument('--units', choices=['V', 'uV'], required=True)
    prediction = commands.add_parser('predict', help='Signal-only checkpoint inference')
    for name in ('checkpoint', 'features', 'output'):
        prediction.add_argument(f'--{name}', type=Path, required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop('command')
    if command == 'run':
        from .workflows.experiment import run
        print(json.dumps(run(arguments['config']), indent=2))
    elif command == 'prepare':
        from .data.manifest import prepare_legacy
        manifest = prepare_legacy(**arguments)
        print(json.dumps({'records': len(manifest['records']), 'endpoint_exclusions': manifest['endpoint_exclusions']}))
    else:
        import torch
        from .evaluation.predict import predict
        torch.set_num_threads(2)
        result = predict(**arguments)
        print(json.dumps({k: list(v.shape) for k, v in result.items()}))
