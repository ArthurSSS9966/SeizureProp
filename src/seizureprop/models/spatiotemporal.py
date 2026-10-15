"""Shared waveform encoder with causal time and equivariant channel attention.

No anatomy, fixed channel ID, clinical onset time, or graph is an encoder input.
The model is a compact experimental architecture, not a SPINT reproduction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderConfig:
    samples_per_patch: int = 256
    width: int = 32
    heads: int = 4
    layers: int = 2
    spatial: bool = True
    dropout: float = 0.1


class FactorizedBlock(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * 2, config.dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.spatial = (nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * 2, config.dropout,
            batch_first=True, norm_first=True, activation="gelu")
            if config.spatial else None)

    def forward(self, h: Tensor) -> Tensor:
        times = h.shape[1]
        causal = torch.ones(times, times, dtype=torch.bool, device=h.device).triu(1)
        h = self.temporal(h, src_mask=causal)
        if self.spatial is not None:
            h = self.spatial(h.transpose(0, 1)).transpose(0, 1)
        return h


class SharedWaveformEncoder(nn.Module):
    """Encode [channels, frames, samples] into [channels, frames, width]."""

    def __init__(self, config: EncoderConfig = EncoderConfig()) -> None:
        super().__init__()
        if config.width % config.heads or config.width % 2 or config.layers < 1:
            raise ValueError("Even width divisible by heads and positive depth required")
        if config.samples_per_patch < 32:
            raise ValueError("At least 32 samples per patch required")
        self.config = config
        self.tokenizer = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(128, config.width))
        self.mask_token = nn.Parameter(torch.zeros(config.width))
        self.blocks = nn.ModuleList([FactorizedBlock(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)

    def forward(self, signal: Tensor, mask: Tensor | None = None) -> Tensor:
        if signal.ndim != 3 or signal.shape[-1] != self.config.samples_per_patch:
            raise ValueError("Expected channels by frames by configured samples")
        channels, frames, samples = signal.shape
        if channels < 1 or frames < 1:
            raise ValueError("Empty channel or time axis")
        if mask is not None:
            if mask.shape != (channels, frames) or mask.dtype != torch.bool:
                raise ValueError("Mask must be boolean channels by frames")
            signal = signal.masked_fill(mask[..., None], 0)
        h = self.tokenizer(signal.reshape(-1, 1, samples)).reshape(channels, frames, -1)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token, h)
        position = torch.arange(frames, device=h.device, dtype=h.dtype)[:, None]
        rates = torch.exp(torch.arange(0, self.config.width, 2, device=h.device,
                                      dtype=h.dtype) * (-math.log(10000) / self.config.width))
        encoding = torch.zeros(frames, self.config.width, device=h.device, dtype=h.dtype)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * rates), torch.cos(position * rates)
        h = h + encoding
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def configuration(self) -> dict:
        return asdict(self.config)


class SpectralReconstruction(nn.Module):
    """Masked log-magnitude spectrum objective; targets are never encoder inputs."""

    def __init__(self, encoder: SharedWaveformEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.config.width, encoder.config.samples_per_patch // 2 + 1)

    def forward(self, signal: Tensor, mask: Tensor) -> Tensor:
        return self.decoder(self.encoder(signal, mask))

    def loss(self, signal: Tensor, mask: Tensor) -> Tensor:
        if not mask.any():
            raise ValueError("At least one masked patch required")
        window = torch.hann_window(signal.shape[-1], device=signal.device)
        target = torch.log1p(torch.fft.rfft(signal * window).abs() / math.sqrt(signal.shape[-1]))
        return (self(signal, mask)[mask] - target.detach()[mask]).square().mean()


class SeparateTaskModel(nn.Module):
    """One task per fitted model; detection labels apply to the multichannel bag."""

    def __init__(self, encoder: SharedWaveformEncoder, task: str, freeze: bool = False) -> None:
        super().__init__()
        if task not in {"detection", "onset", "participation10"}:
            raise ValueError("Unsupported task")
        self.encoder, self.task, self.freeze = encoder, task, freeze
        self.head = nn.Linear(encoder.config.width, 1)
        if freeze:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True) -> SeparateTaskModel:
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, signal: Tensor) -> Tensor:
        h = self.encoder(signal)[:, -1]
        if self.task == "detection":
            h = h.mean(0)  # One global label; no invented per-channel ictal labels.
        return self.head(h).squeeze(-1)
