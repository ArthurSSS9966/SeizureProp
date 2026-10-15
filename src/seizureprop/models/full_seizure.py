"""Independent channel experts over variable-duration seizure recordings."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FullSeizureExpert(nn.Module):
    """Map [channels, seconds, features] to onset or time-resolved activity.

    Each instance has one task and its own parameters. Spatial context is
    permutation equivariant; recurrent temporal context is causal. Onset pooling
    can use the entire supplied recording and does not enforce early decisions.
    """

    def __init__(self, input_dim: int, task: str, contextual: bool = True,
                 width: int = 32) -> None:
        super().__init__()
        if task not in {"onset", "activity"}:
            raise ValueError("One independent onset or activity task is required")
        self.task, self.contextual = task, contextual
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        if contextual:
            self.spatial = nn.MultiheadAttention(width, 4, batch_first=True)
            self.norm = nn.LayerNorm(width)
            self.temporal = nn.GRU(width, width, batch_first=True)
        self.pool = nn.Linear(width, 1) if task == "onset" else None
        self.output = nn.Linear(width, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or not features.shape[1]:
            raise ValueError("Expected nonempty [channels, time, features]")
        h = self.project(features)
        if self.contextual:
            spatial = h.transpose(0, 1)
            message, _ = self.spatial(spatial, spatial, spatial, need_weights=False)
            h = self.norm(h + message.transpose(0, 1))
            h, _ = self.temporal(h)
        if self.task == "onset":
            weights = self.pool(h).softmax(dim=1)
            h = (weights * h).sum(dim=1)
        return self.output(h).squeeze(-1)
