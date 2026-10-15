"""Portable legacy ResWaveNet architecture for matched onset retraining.

Layer names, initial valid-convolution truncation and residual computation match
legacy.models.ResNet. Return the difference between two class logits for BCE;
sigmoid of that difference equals the legacy class-one softmax probability.
No historical checkpoint, optimizer state or patient-specific training is reused.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LegacyResWaveOnset(nn.Module):
    def __init__(self, input_dim: int = 1, kernel_size: int = 16,
                 base_filters: int = 32, n_blocks: int = 3, dropout: float = .2) -> None:
        super().__init__()
        self.feature_embedding = nn.Linear(input_dim, base_filters)
        self.initial = nn.Sequential(nn.Conv1d(base_filters, base_filters, kernel_size),
            nn.BatchNorm1d(base_filters), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'dilated_conv': nn.Conv1d(base_filters, 2*base_filters, kernel_size, dilation=2**i),
            'norm': nn.BatchNorm1d(base_filters),
            'res_conv': nn.Conv1d(base_filters, base_filters, 1),
            'dropout': nn.Dropout(dropout)}) for i in range(n_blocks)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, 2)

    def forward(self, signal: Tensor) -> Tensor:
        """Score [channels, features, samples_or_frames] independently at evaluation."""
        x = self.feature_embedding(signal.transpose(1,2)).transpose(1,2)
        x = self.initial(F.pad(x,(1,0)))
        skip = torch.zeros_like(x)
        for block in self.blocks:
            residual = x
            conv = block['dilated_conv']
            x = conv(F.pad(x,(conv.dilation[0]*(conv.kernel_size[0]-1),0)))
            filt, gate = x.chunk(2,dim=1)
            x = block['dropout'](block['norm'](block['res_conv'](torch.tanh(filt)*torch.sigmoid(gate))))+residual
            skip = skip+x
        logits = self.fc(self.global_pool(skip).squeeze(-1))
        return logits[:,1]-logits[:,0]
