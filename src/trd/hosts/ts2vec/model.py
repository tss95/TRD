"""The evaluated native TS2Vec recipe and its full-window TRD auxiliary."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from trd.features import BackboneOutput
from trd.hosts.ts2vec.vendor.hard_losses import hierarchical_contrastive_loss
from trd.hosts.ts2vec.vendor.encoder import TSEncoder
from trd.loss import TemporalReversalLoss
from trd.random import preserve_rng
from trd.targets import sample_reversal


class TS2Vec(nn.Module):
    """Train the live encoder and expose the averaged encoder to frozen probes."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        settings = config['encoder']
        self.channels = config['channels']
        self.embedding_dim = settings['output_dims']
        self.temporal_unit = settings['temporal_unit']
        self.lambda_ = settings['lambda_']
        if settings['tau_temp'] != 0:
            raise ValueError('This release includes the native TS2Vec objective only')
        self.trd_weight = config['trd']['weight']
        self.span, self.fade = config['trd']['span_samples'], config['trd']['fade_samples']
        self._encoder = TSEncoder(
            self.channels, self.embedding_dim, settings['hidden_dims'], settings['depth'], settings['mask_mode']
        )
        self._encoder_swa = AveragedModel(self._encoder)
        self._encoder_swa.update_parameters(self._encoder)
        self._encoder_swa.requires_grad_(False).eval()
        self.trd = None
        if self.trd_weight > 0:
            with torch.random.fork_rng(devices=[]):
                self.trd = TemporalReversalLoss(self.embedding_dim)

    def train(self, mode: bool = True) -> TS2Vec:
        """Keep the averaged encoder in evaluation mode while fitting the live one."""
        super().train(mode)
        self._encoder_swa.eval()
        return self

    @staticmethod
    def _take_per_row(x: torch.Tensor, start: torch.Tensor, length: int) -> torch.Tensor:
        offsets = start[:, None] + torch.arange(length, device=x.device)[None, :]
        return x[torch.arange(x.size(0), device=x.device)[:, None], offsets]

    def _crops(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        minimum = 2 ** (self.temporal_unit + 1)
        if length < minimum:
            raise ValueError('Waveform is shorter than the native contrastive support')
        crop = int(torch.randint(minimum, length + 1, ()).item())
        left = int(torch.randint(0, length - crop + 1, ()).item())
        right = left + crop
        extended_left = int(torch.randint(0, left + 1, ()).item())
        extended_right = int(torch.randint(right, length + 1, ()).item())
        offsets = torch.randint(-extended_left, length - extended_right + 1, (batch,), device=x.device)
        first = self._take_per_row(x, offsets + extended_left, right - extended_left)
        second = self._take_per_row(x, offsets + left, extended_right - left)
        first_features = self._encoder(first, mask=None)
        second_features = self._encoder(second, mask=None)
        return first_features[:, -crop:], second_features[:, :crop]

    def forward(self, x: torch.Tensor, second: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute native loss and optional reversal loss from a BTC source window."""
        del second
        if x.ndim != 3 or x.shape[-1] != self.channels or not x.is_floating_point():
            raise ValueError(f'Expected floating point [B,T,{self.channels}] input')
        first, second_features = self._crops(x)
        native = hierarchical_contrastive_loss(
            first, second_features, alpha=self.lambda_, temporal_unit=self.temporal_unit
        )
        total = native
        auxiliary = native.new_zeros(())
        if self.trd is not None:
            with preserve_rng():
                corrupted, targets, valid, _ = sample_reversal(x, self.span, self.fade)
                features = self._encoder(corrupted, mask='all_true')
                auxiliary = self.trd(features, targets, valid)
            total = native + self.trd_weight * auxiliary
        return {'total_loss': total, 'native_loss': native, 'trd_loss': auxiliary}

    def after_optimizer_step(self) -> None:
        """Update the same running parameter average used by the paper readouts."""
        self._encoder_swa.update_parameters(self._encoder)

    def encode(self, x: torch.Tensor) -> BackboneOutput:
        """Return final unpooled averaged features on clean BTC inputs."""
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f'Expected [B,T,{self.channels}] input')
        valid = ~x.isnan().any(dim=-1)
        tokens = self._encoder_swa(x.clone(), mask='all_true')
        return BackboneOutput(tokens=tokens, states=None, cls=None, mask=valid, stride=1)
