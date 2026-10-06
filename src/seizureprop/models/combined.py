"""Causal multi-head backbone; legacy state-dict names are preserved."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class CombinedScorer(nn.Module):
    """Shared causal backbone with independent activity/localization/tissue heads.

    All ablations instantiate the same model in the same order. The tissue
    head pools baseline windows only. Localization does not require a tissue
    label or the tissue prediction to produce scores.
    """
    def __init__(self, features: int, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Conv1d(features, hidden, 1)
        self.layers = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=d) for d in (1, 2)])
        self.dropout = nn.Dropout(.1)
        self.localization_head = nn.Conv1d(hidden, 1, 1)
        self.activity_head = nn.Conv1d(hidden, 1, 1)
        self.tissue_head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, nbase: int) -> dict[str, Tensor]:
        if not 0 < nbase < x.shape[-1]:
            raise ValueError("Supply a nonempty baseline and ictal interval")
        hidden = F.gelu(self.embed(x))
        for layer in self.layers:
            hidden = hidden + self.dropout(F.gelu(layer(F.pad(hidden, (2 * layer.dilation[0], 0)))))
        return {
            "localization": self.localization_head(hidden).squeeze(1),
            "activity": self.activity_head(hidden).squeeze(1),
            "tissue": self.tissue_head(hidden[:, :, :nbase].mean(-1)).squeeze(-1),
        }
