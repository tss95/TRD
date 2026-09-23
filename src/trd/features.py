"""Features and explicit temporal coordinates shared by the readouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ScaleLevel:
    """A single multi-scale representation level exposed by a backbone."""

    s: int
    repr: torch.Tensor
    stride: int | None = None
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class BackboneOutput:
    """
    Canonical backbone outputs used by downstream tasks.

    Semantic intent:
      - tokens: local, highest-resolution sequence features  [B, T, D]
      - states: medium-range sequence/window features        [B, W, D] (shape may vary by backbone)
      - cls:    global, whole-series representation          [B, D]
      - mask:   valid-token mask for tokens; True means valid [B, T]
    """

    tokens: torch.Tensor | None
    states: torch.Tensor | None
    cls: torch.Tensor | None
    mask: torch.Tensor | None
    stride: int | None
    extras: dict[str, Any] = field(default_factory=dict)
    scales: tuple[ScaleLevel, ...] | None = None
    # Optional absolute sample centers [T], shared by the batch. Existing
    # backbones retain their current uniform-grid interpretation when absent.
    sample_positions: torch.Tensor | None = None

    def global_features(self, *, pool: str = "default") -> torch.Tensor:
        """Return one global feature vector per sample, shape `[B, D]`."""

        del pool
        if self.cls is not None:
            return self.cls
        if self.states is not None:
            states = self.states
            if states.dim() == 4:
                return states.mean(dim=(1, 2))
            if states.dim() == 3:
                return states.mean(dim=1)
            if states.dim() == 2:
                return states
        if self.tokens is not None:
            return self.tokens.mean(dim=1)
        raise RuntimeError("BackboneOutput cannot produce global features from empty tokens/states/cls.")

    def sequence_features(self) -> torch.Tensor:
        """Return high-resolution sequence features, shape `[B, T, D]`."""

        if self.tokens is not None:
            if self.tokens.dim() != 3:
                raise RuntimeError(f"BackboneOutput tokens must have shape [B,T,D], got {tuple(self.tokens.shape)}.")
            return self.tokens
        if self.states is not None and self.states.dim() == 3:
            return self.states
        raise RuntimeError("BackboneOutput cannot produce sequence features without 3D tokens or states.")

    def per_window_features(self) -> torch.Tensor:
        """Return per-window features with any slot axis pooled, shape `[B, W, D]`."""

        if self.states is not None:
            if self.states.dim() == 4:
                return self.states.mean(dim=2)
            if self.states.dim() == 3:
                return self.states
            if self.states.dim() == 2:
                return self.states.unsqueeze(1)
        if self.tokens is not None:
            return self.sequence_features()
        raise RuntimeError("BackboneOutput cannot produce per-window features from empty states/tokens.")
