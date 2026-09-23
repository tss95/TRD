"""The paper's affine interval-membership head and unweighted binary loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalReversalLoss(nn.Module):
    """Predict interval membership from final encoder tokens with one affine map."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError('embedding_dim must be positive')
        self.head = nn.Linear(embedding_dim, 1)

    def compute_loss(
        self, tokens: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Average BCE over valid tokens, preserving the principal host reduction."""
        if tokens.ndim != 3 or labels.shape != tokens.shape[:2] or tokens.numel() == 0:
            raise ValueError('Expected nonempty tokens [B,T,D] and labels [B,T]')
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape != labels.shape or not bool(valid.any()):
                raise ValueError('Expected a nonempty boolean validity mask [B,T]')
            tokens, labels = tokens[valid].unsqueeze(0), labels[valid].unsqueeze(0)
        labels = labels.float()
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError('TRD membership targets must be binary')
        logits = self.head(tokens).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite TRD loss')
        return {
            'total_loss': loss,
            'acc': ((logits.detach() > 0) == labels.bool()).float().mean(),
            'pos_frac': labels.mean(),
        }

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """Return the differentiable auxiliary loss."""
        return self.compute_loss(tokens, labels, valid)['total_loss']
