"""Per-channel experts with optional soft evidence and masked supervision."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spatiotemporal import SharedWaveformEncoder


def masked_channel_bce(logits: Tensor, target: Tensor) -> Tensor:
    """Balance known positive/negative classes; negative/NaN targets are unknown.

    Unknown contacts contribute neither a gradient nor a class-weight count.
    The caller must skip samples without any usable labels.
    """
    if logits.shape != target.shape:
        raise ValueError("Logits and target shapes must agree")
    known = torch.isfinite(target) & (target >= 0)
    if not known.any():
        raise ValueError("No annotated contacts in this sample")
    labels, scores = target[known], logits[known]
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Known labels must be binary")
    loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    return torch.stack([loss[labels == k].mean() for k in (0, 1)
                        if (labels == k).any()]).mean()


def participation_features(probabilities: Tensor) -> Tensor:
    """Summarize [channels, ordered observations] without hard channel selection.

    Every observation must be available by the requested prediction time. The
    caller is responsible for patient-excluded fitting and temporal provenance.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("At least two ordered observations per channel required")
    if not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Finite sigmoid scores in [0, 1] required")
    return torch.stack((probabilities[:, -1], probabilities.mean(1),
                        probabilities[:, -1] - probabilities[:, 0],
                        probabilities.amax(1)), dim=-1)


class ChannelExpert(nn.Module):
    """One binary score per channel, optionally informed by four soft features.

    Temporal summaries retain the final representation, context mean, and change
    across the context. Scores are logits; sigmoid is applied for display and
    thresholding. They are not calibrated probabilities by construction.
    """

    def __init__(self, encoder: SharedWaveformEncoder, auxiliary: bool = False,
                 linear: bool = False) -> None:
        super().__init__()
        self.encoder, self.auxiliary, self.linear = encoder, auxiliary, linear
        width = encoder.config.width
        inputs = width if linear else 3 * width + (4 if auxiliary else 0)
        self.head = (nn.Linear(inputs, 1) if linear else
                     nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, 1)))
        if linear and auxiliary:
            raise ValueError("Linear control does not accept auxiliary evidence")

    def forward(self, signal: Tensor, auxiliary: Tensor | None = None) -> Tensor:
        h = self.encoder(signal)
        features = h[:, -1] if self.linear else torch.cat(
            (h[:, -1], h.mean(1), h[:, -1] - h[:, 0]), dim=-1)
        if self.auxiliary:
            if auxiliary is None or auxiliary.shape != (h.shape[0], 4):
                raise ValueError("Four auxiliary values per channel required")
            if not torch.isfinite(auxiliary).all():
                raise ValueError("Nonfinite auxiliary evidence")
            features = torch.cat((features, auxiliary.detach()), dim=-1)
        elif auxiliary is not None:
            raise ValueError("Unexpected auxiliary evidence")
        return self.head(features).squeeze(-1)
