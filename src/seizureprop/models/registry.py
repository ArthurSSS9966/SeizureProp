"""Explicit architecture selection, independent of objectives and devices."""
import torch
from torch import Tensor, nn
from .combined import CombinedScorer


class ContextScorer(CombinedScorer):
    """Compare each channel with the contemporaneous signal ensemble.

    The added branch is causal and permutation equivariant. Zero initialization
    preserves the pretrained backbone's predictions exactly at epoch zero.
    It uses no channel names, coordinates or tissue annotations.
    """

    def __init__(self, features: int, hidden: int = 32) -> None:
        super().__init__(features, hidden)
        self.context_head = nn.Conv1d(features, 1, 1, bias=False)
        nn.init.zeros_(self.context_head.weight)

    def forward(self, x: Tensor, nbase: int) -> dict[str, Tensor]:
        output = super().forward(x, nbase)
        contrast = x - x.mean(0, keepdim=True)
        output['localization'] = output['localization'] + self.context_head(torch.tanh(contrast)).squeeze(1)
        return output


def build_model(name: str, features: int, hidden: int) -> nn.Module:
    registry = {'combined': CombinedScorer, 'context': ContextScorer}
    if name not in registry:
        raise ValueError(f'Unknown model: {name}')
    return registry[name](features, hidden)


def initialize(model: nn.Module, state: dict) -> None:
    """Allow only the documented, zero-initialized context extension."""
    result = model.load_state_dict(state, strict=False)
    allowed = {'context_head.weight'} if isinstance(model, ContextScorer) else set()
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise ValueError(f'Incompatible initialization: {result}')
