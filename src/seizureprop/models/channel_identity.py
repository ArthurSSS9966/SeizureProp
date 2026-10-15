"""Compact signal-derived channel identity with equivariant population attention.

Inspired by SPINT's calibration principle, not an implementation of SPINT.
The temporal convolutions are causal; population attention mixes channels only
at the same time. Baseline calibration is available before seizure inference.
No channel identifiers, tissue annotations or coordinates enter this module.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChannelIdentityScorer(nn.Module):
    """Decode individual channels through a small shared population bottleneck."""

    def __init__(self, features: int, hidden: int = 32, identity: bool = True) -> None:
        super().__init__()
        if hidden < 4 or hidden % 4:
            raise ValueError("Attention width must be a positive multiple of four")
        self.use_identity = identity
        self.embed = nn.Conv1d(features, hidden, 1)
        self.layers = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 3, dilation=d) for d in (1, 2)
        ])
        self.id_frames = nn.Sequential(nn.Conv1d(features, hidden, 1), nn.GELU())
        self.id_pool = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                     nn.LayerNorm(hidden))
        self.latents = nn.Parameter(torch.randn(8, hidden) * .02)
        self.population_read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.channel_read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                nn.Linear(2 * hidden, hidden))
        self.dropout = nn.Dropout(.1)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden))
        self.localization_head = nn.Conv1d(hidden, 1, 1)
        self.reconstruction_head = nn.Conv1d(hidden, features // 2, 1)

    def channel_identity(self, calibration: Tensor) -> Tensor:
        """Infer one vector per channel from explicitly supplied baseline only."""
        if calibration.ndim != 3 or calibration.shape[-1] < 1:
            raise ValueError("Nonempty channel-by-feature-by-time calibration required")
        frames = self.id_frames(calibration)
        pooled = torch.cat([frames.mean(-1), frames.std(-1, unbiased=False)], dim=1)
        identity = self.id_pool(pooled)
        return identity if self.use_identity else torch.zeros_like(identity)

    def encode(self, x: Tensor, calibration: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode signals, masking whole feature vectors before any convolution.

        Args:
            x: Channels by features by time.
            calibration: Disjoint baseline for reconstruction, or reserved
                pre-onset baseline for localization.
            mask: Optional channels by time mask, True for hidden tokens.
        """
        if mask is not None:
            if mask.shape != (x.shape[0], x.shape[-1]) or mask.dtype != torch.bool:
                raise ValueError("Mask must be boolean channels by time")
            x = x.masked_fill(mask[:, None, :], 0)
        h = F.gelu(self.embed(x))
        if mask is not None:
            h = h + mask[:, None, :] * self.mask_embedding[None, :, None]
        for layer in self.layers:
            h = h + self.dropout(F.gelu(layer(F.pad(h, (2 * layer.dilation[0], 0)))))
        h = h + self.channel_identity(calibration)[:, :, None]
        # Time is the batch dimension: attention never mixes future timepoints.
        tokens = h.permute(2, 0, 1)
        queries = self.latents[None].expand(tokens.shape[0], -1, -1)
        population, _ = self.population_read(queries, tokens, tokens, need_weights=False)
        context, _ = self.channel_read(tokens, population, population, need_weights=False)
        tokens = self.norm(tokens + self.dropout(context))
        tokens = tokens + self.dropout(self.ff(tokens))
        return tokens.permute(1, 2, 0)

    def forward(self, x: Tensor, nbase: int) -> dict[str, Tensor]:
        if not 2 < nbase < x.shape[-1]:
            raise ValueError("At least three baseline frames and one ictal frame required")
        # Avoid the final two pre-onset seconds in identity calibration.
        h = self.encode(x, x[:, :, :nbase - 2])
        return {"localization": self.localization_head(h).squeeze(1)}

    def reconstruct(self, x: Tensor, calibration: Tensor, mask: Tensor) -> Tensor:
        """Predict absolute signal descriptors of masked tokens, without labels."""
        return self.reconstruction_head(self.encode(x, calibration, mask))


class ChannelAttentionScorer(ChannelIdentityScorer):
    """Matched architecture with baseline identity disabled."""

    def __init__(self, features: int, hidden: int = 32) -> None:
        super().__init__(features, hidden, identity=False)
