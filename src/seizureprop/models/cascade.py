"""Independent event detection and activity-informed onset models."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from seizureprop.models.full_seizure import FullSeizureExpert


class EventDetector(nn.Module):
    """Causal channel-invariant detector with a 15-frame receptive field.

    Input is [channels, frames, features]; output is one logit per frame.
    Learned attention and maximum pooling retain evidence from focal channels.
    No absolute time, channel count threshold or clinical boundary is an input.
    """

    def __init__(self, input_dim: int = 17, width: int = 32) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.attention = nn.Linear(width, 1)
        self.mix = nn.Linear(width * 2, width)
        self.temporal = nn.ModuleList([
            nn.Conv1d(width, width, 3, dilation=d) for d in (1, 2, 4)])
        self.output = nn.Linear(width, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or min(features.shape[:2]) < 1:
            raise ValueError("Expected nonempty channels by frames by features")
        h = self.project(features)
        weights = self.attention(h).softmax(dim=0)
        pooled = torch.cat([(weights * h).sum(0), h.amax(0)], dim=-1)
        h = self.mix(pooled).T.unsqueeze(0)
        for layer in self.temporal:
            h = h + F.gelu(layer(F.pad(h, (2 * layer.dilation[0], 0))))
        return self.output(h.squeeze(0).T).squeeze(-1)


class ActivityInformedOnset(nn.Module):
    """Long-sequence onset classifier with optional soft upstream evidence.

    All channels remain available, including low-activity channels. Predicted
    activity is a feature, never a hard mask or a mandatory score multiplier.
    No gradient is propagated into the independently trained activity expert.
    """

    def __init__(self, input_dim: int = 17, use_activity: bool = True,
                 width: int = 32) -> None:
        super().__init__()
        self.use_activity = use_activity
        self.expert = FullSeizureExpert(input_dim + int(use_activity), "onset",
                                       contextual=True, width=width)

    def forward(self, features: Tensor, activity: Tensor | None = None) -> Tensor:
        if self.use_activity:
            if activity is None or activity.shape != features.shape[:2]:
                raise ValueError("Aligned channel-time activity probabilities required")
            if not torch.isfinite(activity).all() or ((activity < 0) | (activity > 1)).any():
                raise ValueError("Activity probabilities must be finite and in [0, 1]")
            features = torch.cat([features, activity.detach().unsqueeze(-1)], dim=-1)
        return self.expert(features)
