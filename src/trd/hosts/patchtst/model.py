"""Fixed-peak masked reconstruction and optional fused TRD."""

from __future__ import annotations
from typing import Any
import torch
from torch import nn
from trd.config import namespace
from trd.features import BackboneOutput
from trd.hosts.patchtst.backbone import PatchTSTSSLBackbone
from trd.hosts.patchtst.auxiliary import PatchTSTTRD, install_view_batchnorm, batchnorm_view_context
from trd.hosts.patchtst.vendor.patch_mask import create_patch, random_masking


class PatchTST(nn.Module):
    """The paper's single-device PatchTST training and feature interfaces."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = namespace(
            {
                'model': {'patchtst_ssl': config['encoder']},
                'data_parameters': {'num_channels': config['channels']},
                'contrast_parameters': {
                    'temporal_reversal': {
                        'mode': 'segment',
                        'span_tokens': config['trd']['span_samples'] // config['encoder']['stride'],
                        'crossfade_samples': config['trd']['fade_samples'],
                    }
                },
            }
        )
        if config['trd']['span_samples'] % config['encoder']['stride']:
            raise ValueError('The paper PatchTST span must be a stride multiple')
        self.encoder = PatchTSTSSLBackbone(self.cfg)
        self.embedding_dim = self.encoder.embedding_dim
        self.lambda_trd = config['trd']['weight']
        self.trd = PatchTSTTRD(self.cfg, embedding_dim=self.embedding_dim) if self.lambda_trd > 0 else None
        if self.trd is not None:
            install_view_batchnorm(self.encoder.model.backbone)

    def forward(self, x: torch.Tensor, second: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute reconstruction and optional membership loss on BTC windows."""
        del second
        context, valid = self.training_context(x.transpose(1, 2))
        if self.trd is not None:
            _, logs = self._forward_with_trd(context, valid)
            return logs
        loss = self.reconstruction_loss(context, valid)
        return {'total_loss': loss, 'patchtst_reconstruction_loss': loss}

    def encode(self, x: torch.Tensor) -> BackboneOutput:
        """Extract clean final patch features on their retained temporal grid."""
        return self.encoder.forward_backbone_output(x, input_layout='BTC')

    def training_context(self, x_bct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a context within each recording; pad short recordings explicitly."""
        x = x_bct.transpose(1, 2)
        self.encoder.validate_input(x)
        batch, length, _ = x.shape
        context = self.encoder.context_length
        if length == context:
            return x, torch.ones(batch, context, device=x.device, dtype=torch.bool)
        if length > context:
            starts = torch.randint(length - context + 1, (batch,), device=x.device)
            positions = starts[:, None] + torch.arange(context, device=x.device)
            return x[torch.arange(batch, device=x.device)[:, None], positions], torch.ones(
                batch, context, device=x.device, dtype=torch.bool
            )
        left = min(self.encoder.patch_origin, context - length)
        padded = x.new_zeros(batch, context, self.encoder.channels)
        padded[:, left : left + length] = x
        valid = torch.zeros(batch, context, device=x.device, dtype=torch.bool)
        valid[:, left : left + length] = True
        return padded, valid

    def reconstruction_loss(self, x_btc: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """Compute source MSE on masked patches, excluding padding from targets."""
        target, validity = self.encoder.normalized_patches(x_btc, valid)
        masked, mask = self._mask_patches(target, validity)
        prediction = self.encoder.model(masked, key_padding_mask=self.encoder.key_padding_mask(validity))
        return self._reconstruction_mse(prediction, target, validity, mask)

    def _mask_patches(self, target: torch.Tensor, validity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(validity.all()):
            masked, _, mask, _ = random_masking(target, self.cfg.model.patchtst_ssl.mask_ratio)
        else:
            # Ragged contexts mask only real patch slots. Each channel retains
            # its own source permutation; the original temporal positions stay fixed.
            masked = target.clone()
            mask = target.new_zeros(target.shape[:3])
            for row in range(target.shape[0]):
                for channel in range(target.shape[2]):
                    slots = validity[row, :, channel].any(dim=-1).nonzero().flatten()
                    if slots.numel() == 0:
                        raise ValueError("PatchTST training context has no valid retained patch")
                    values = target[row, slots, channel][None, :, None, :]
                    values, _, selected, _ = random_masking(values, self.cfg.model.patchtst_ssl.mask_ratio)
                    masked[row, slots, channel] = values[0, :, 0]
                    mask[row, slots, channel] = selected[0, :, 0]
        return masked, mask

    @staticmethod
    def _reconstruction_mse(
        prediction: torch.Tensor, target: torch.Tensor, validity: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        patch_loss = ((prediction - target).square() * validity).sum(dim=-1)
        patch_loss = patch_loss / validity.sum(dim=-1).clamp_min(1)
        loss = (patch_loss * mask).sum() / mask.sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PatchTST reconstruction loss is not finite")
        return loss

    def _forward_with_trd(
        self, context: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode reconstruction/reversal views with separate normalization groups."""
        if self.trd is None:
            raise RuntimeError("PatchTST TRD forward requires its auxiliary head")
        normalized, valid = self.encoder.normalized_context(context, valid)
        target, _ = create_patch(normalized, self.encoder.patch_length, self.encoder.stride)
        validity, _ = create_patch(valid[:, :, None], self.encoder.patch_length, self.encoder.stride)
        validity = validity.expand(-1, -1, self.encoder.channels, -1)
        masked, mask = self._mask_patches(target, validity)
        reversed_patches, labels, label_valid, _ = self.trd.prepare(normalized, valid)
        key_mask = self.encoder.key_padding_mask(validity)
        backbone = self.encoder.model.backbone
        batch = context.size(0)
        if self.cfg.model.patchtst_ssl.fuse_trd_views:
            fused_mask = None if key_mask is None else torch.cat((key_mask, key_mask), dim=0)
            with batchnorm_view_context(backbone, view_batch_size=batch * self.encoder.channels):
                encoded = backbone(torch.cat((masked, reversed_patches), dim=0), key_padding_mask=fused_mask)
            reconstruction_features, trd_features = encoded.split(batch, dim=0)
        else:
            reconstruction_features = backbone(masked, key_padding_mask=key_mask)
            with batchnorm_view_context(backbone, update_running_stats=False):
                trd_features = backbone(reversed_patches, key_padding_mask=key_mask)
        prediction = self.encoder.model.head(reconstruction_features)
        reconstruction = self._reconstruction_mse(prediction, target, validity, mask)
        trd_logs = self.trd.compute_loss(trd_features, labels, label_valid)
        total = reconstruction + self.lambda_trd * trd_logs["total_loss"]
        return total, {
            "total_loss": total,
            "patchtst_reconstruction_loss": reconstruction,
            "trd_loss": trd_logs["total_loss"],
            "trd_acc": trd_logs["acc"],
            "trd_pos_frac": trd_logs["pos_frac"],
        }
