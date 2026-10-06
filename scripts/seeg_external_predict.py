"""Compatibility command; install the local package with pip install -e . first."""
import argparse
from pathlib import Path
from seizureprop.evaluation.predict import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'features', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    import torch
    torch.set_num_threads(4)
    predict(**vars(parser.parse_args()))
