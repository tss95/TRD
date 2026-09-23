"""Dense adapter for the released channel-independent PatchTST SSL encoder."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import nn

from trd.layout import convert_input_layout
from trd.features import BackboneOutput
from trd.hosts.patchtst.vendor.models.patchTST import PatchTST
from trd.hosts.patchtst.vendor.models.layers.revin import RevIN
from trd.hosts.patchtst.vendor.patch_mask import create_patch

SOURCE_REVISION = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


class PatchTSTSSLBackbone(nn.Module):
    """Expose final patch features, with channels concatenated at each timestamp.

    The source encoder is shared across channels. Training receives one native
    context; inference averages overlapping contexts on a common patch grid.
    Padding is zero in normalized space, excluded from normalization and masked
    as attention keys when an entire patch is invalid. Partial patches remain
    usable and their valid sample counts are retained.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.settings = cfg.model.patchtst_ssl
        self.input_preprocessing = {}
        p = self.settings
        self.channels = int(cfg.data_parameters.num_channels)
        self.context_length = int(p.context_length)
        self.patch_length = int(p.patch_length)
        self.stride = int(p.stride)
        if self.channels < 1 or self.patch_length < 1 or self.context_length < self.patch_length:
            raise ValueError("PatchTST SSL requires positive channels and context_length >= patch_length > 0")
        if not 0 < self.stride <= self.patch_length:
            raise ValueError("PatchTST SSL requires 0 < stride <= patch_length; gapped patch grids are unsupported")
        if self.stride < self.patch_length and p.recipe != "sensitivity":
            raise ValueError("Overlapping PatchTST SSL patches require an explicit sensitivity recipe")
        self.num_patches = (self.context_length - self.patch_length) // self.stride + 1
        self.covered_length = self.patch_length + (self.num_patches - 1) * self.stride
        self.patch_origin = self.context_length - self.covered_length
        # Each context contributes num_patches output slots. Overlapping raw
        # supports extend beyond these slots but cannot replace missing slots.
        self.grid_coverage_length = self.num_patches * self.stride
        if int(p.tile_step) <= 0 or int(p.tile_step) > self.grid_coverage_length or int(p.tile_step) % self.stride:
            raise ValueError("PatchTST tile_step must be a positive patch-stride multiple <= covered patch grid")
        if int(p.tile_batch_size) < 1:
            raise ValueError("PatchTST tile_batch_size must be positive")
        if p.probe_input_layout not in {"BTC", "BCT"}:
            raise ValueError("PatchTST probe_input_layout must explicitly be BTC or BCT")
        if p.channel_fusion != "concat":
            raise ValueError("PatchTST SSL currently exposes only explicit fixed channel concatenation")
        if p.normalization not in {"train_standard_revin", "revin_only"}:
            raise ValueError("PatchTST normalization must be train_standard_revin or the revin_only sensitivity")
        if p.source_revision != SOURCE_REVISION:
            raise ValueError("PatchTST source_revision does not identify the vendored implementation")
        if int(p.d_model) < 1 or int(p.n_heads) < 1 or int(p.d_model) % int(p.n_heads):
            raise ValueError("PatchTST d_model must be positive and divisible by n_heads")
        if int(p.n_layers) < 1 or int(p.d_ff) < 1 or not math.isfinite(float(p.revin_eps)) or p.revin_eps <= 0:
            raise ValueError("PatchTST requires positive depth, feed-forward width and finite RevIN epsilon")
        self.embedding_dim = self.channels * int(p.d_model)
        self.model = PatchTST(
            c_in=self.channels,
            target_dim=0,
            patch_len=self.patch_length,
            stride=self.stride,
            num_patch=self.num_patches,
            n_layers=int(p.n_layers),
            n_heads=int(p.n_heads),
            d_model=int(p.d_model),
            shared_embedding=True,
            d_ff=int(p.d_ff),
            dropout=float(p.dropout),
            head_dropout=float(p.head_dropout),
            act="relu",
            head_type="pretrain",
            res_attention=False,
        )
        self.revin = RevIN(self.channels, eps=float(p.revin_eps), affine=False)
        self.register_buffer("standard_mean", torch.zeros(self.channels))
        self.register_buffer("standard_std", torch.ones(self.channels))
        self.register_buffer("standard_fitted", torch.tensor(p.normalization == "revin_only"))
        self.register_buffer("selected_peak_lr", torch.tensor(0.0, dtype=torch.float64))

    @property
    def backbone_input_layout(self) -> str:
        """Declare the direct-call layout to the shared downstream dispatcher."""
        return self.settings.probe_input_layout

    def validate_input(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """Validate explicit BTC input and return its True=valid sample mask."""
        if x.ndim != 3 or x.shape[-1] != self.channels or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"PatchTST expects nonempty [B,T,{self.channels}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("PatchTST requires floating-point inputs")
        if valid is None:
            valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        if valid.dtype != torch.bool or valid.shape != x.shape[:2] or valid.device != x.device:
            raise ValueError("PatchTST valid mask must be bool [B,T] on the input device")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("PatchTST received an all-invalid sample")
        if not bool(torch.isfinite(x.masked_fill(~valid.unsqueeze(-1), 0)).all()):
            raise ValueError("PatchTST requires finite valid samples; provide an explicit mask for missing values")
        return valid

    def normalized_patches(
        self, x: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply train-fitted standard scaling, RevIN, and source tail patching."""
        x, valid = self.normalized_context(x, valid)
        patches, _ = create_patch(x, self.patch_length, self.stride)
        validity, _ = create_patch(valid.unsqueeze(-1), self.patch_length, self.stride)
        return patches, validity.expand(-1, -1, self.channels, -1)

    def normalized_context(
        self, x: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized waveform before patching or TRD corruption.

        All-valid native contexts use the unmodified source RevIN. For padded
        contexts the same population moments exclude invalid samples.
        """
        valid = self.validate_input(x, valid)
        if x.shape[1] != self.context_length:
            raise ValueError(f"PatchTST native context must have length {self.context_length}")
        if not bool(self.standard_fitted):
            raise RuntimeError("PatchTST standard scaler is unfitted; prepare training or load a fitted checkpoint")
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        x = (x - self.standard_mean) / self.standard_std
        if bool(valid.all()):
            x = self.revin(x, "norm")
        else:
            count = valid.sum(dim=1, keepdim=True).unsqueeze(-1)
            mean = (x * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / count
            var = ((x - mean).square() * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / count
            x = (x - mean.detach()) / (var + self.settings.revin_eps).sqrt().detach()
            x = x.masked_fill(~valid.unsqueeze(-1), 0)
        return x, valid

    @staticmethod
    def key_padding_mask(valid_patches: torch.Tensor) -> torch.Tensor | None:
        """Convert [B,P,C,K] validity to source attention's [B*C,P] exclusion mask."""
        keep = valid_patches.any(dim=-1).transpose(1, 2).flatten(0, 1)
        if not bool(keep.any(dim=1).all()):
            raise ValueError("PatchTST source tail patching leaves a channel with no valid patch")
        return None if bool(keep.all()) else ~keep

    def native_features(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """Return final [B,C,D,P] source features before the reconstruction head."""
        patches, validity = self.normalized_patches(x, valid)
        return self.model.backbone(patches, key_padding_mask=self.key_padding_mask(validity))

    def feature_provenance(self) -> dict[str, Any]:
        """Return the extraction identity used by reports and feature caches."""
        settings = vars(self.settings).copy()
        # Persistence policy does not change features or the historical identity
        # stored in existing PatchTST one-cycle checkpoints.
        settings.pop("full_training_checkpoint", None)
        return {
            "method": "patchtst_ssl",
            "source_revision": SOURCE_REVISION,
            "feature_tap": "model.backbone.final",
            "channels": self.channels,
            "feature_dim": self.embedding_dim,
            "settings": settings,
            "domain_preprocessing": self.input_preprocessing,
            "padding": "zero_after_valid_only_normalization",
            "stitching": "mean_overlapping_patches",
            "geometry_version": 1,
        }

    def feature_cache_identity(self) -> str:
        """Hash recipe, source, normalization, channel fusion and extraction geometry."""
        raw = json.dumps(self.feature_provenance(), sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def forward_backbone_output(
        self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None, input_layout: str | None = None
    ) -> BackboneOutput:
        """Tile without resampling physical time and retain absolute patch centers.

        Tile zero starts at -patch_origin: the source's discarded prefix is
        padding, so the first retained patch starts at sample zero. Subsequent
        origins preserve that grid. Channels are fused only after encoding.
        """
        layout = self.settings.probe_input_layout if input_layout is None else input_layout
        x = convert_input_layout(x, source=layout, target="BTC")
        valid = self.validate_input(x, valid_mask)
        batch, length, _ = x.shape
        n_tokens = math.ceil(length / self.stride)
        n_tiles = 1 + math.ceil(max(0, length - self.grid_coverage_length) / int(self.settings.tile_step))
        starts = torch.arange(n_tiles, device=x.device) * int(self.settings.tile_step) - self.patch_origin
        features = x.new_zeros(batch, n_tokens, self.embedding_dim)
        counts = x.new_zeros(batch, n_tokens, 1)
        # Process bounded tile batches; no full-record attention or view cache.
        for offset in range(0, n_tiles, int(self.settings.tile_batch_size)):
            origins = starts[offset : offset + int(self.settings.tile_batch_size)]
            positions = origins[:, None] + torch.arange(self.context_length, device=x.device)
            in_bounds = (positions >= 0) & (positions < length)
            safe_positions = positions.clamp(0, length - 1)
            windows = x[:, safe_positions].flatten(0, 1)
            window_valid = (valid[:, safe_positions] & in_bounds).flatten(0, 1)
            has_patches = window_valid[:, self.patch_origin :].any(dim=1)
            if not bool(has_patches.any()):
                continue
            patches, patch_valid = self.normalized_patches(windows[has_patches], window_valid[has_patches])
            encoded = self.model.backbone(patches, key_padding_mask=self.key_padding_mask(patch_valid))
            fused = encoded.permute(0, 3, 1, 2).flatten(2)
            weights = patch_valid.any(dim=-1).any(dim=-1).unsqueeze(-1)
            all_features = fused.new_zeros(batch * len(origins), self.num_patches, self.embedding_dim)
            all_weights = fused.new_zeros(batch * len(origins), self.num_patches, 1)
            all_features[has_patches] = fused * weights
            all_weights[has_patches] = weights.to(fused.dtype)
            indices = (origins[:, None] + self.patch_origin) // self.stride
            indices = indices + torch.arange(self.num_patches, device=x.device)
            keep = indices.flatten() < n_tokens
            indices = indices.flatten()[keep]
            contributions = all_features.reshape(batch, -1, self.embedding_dim)[:, keep]
            contributions_weight = all_weights.reshape(batch, -1, 1)[:, keep]
            features = features.index_add(1, indices, contributions.to(features.dtype))
            counts = counts.index_add(1, indices, contributions_weight.to(counts.dtype))
        token_mask = counts.squeeze(-1) > 0
        features = features / counts.clamp_min(1)
        centers = torch.arange(n_tokens, device=x.device, dtype=torch.float32) * self.stride
        # The partially padded last patch is anchored at its actual valid span.
        centers = centers + (torch.minimum(length - centers, centers.new_tensor(self.patch_length)) - 1) / 2
        pooled = features.sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return BackboneOutput(
            tokens=features,
            states=None,
            cls=pooled,
            mask=token_mask,
            stride=self.stride,
            sample_positions=centers,
            extras={
                "input_length": length,
                "tile_starts": starts,
                "patch_origin": self.patch_origin,
                "context_length": self.context_length,
                "channel_fusion": "concat",
                "feature_provenance": self.feature_provenance(),
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense tokens through the explicit configured input layout."""
        return self.forward_backbone_output(x).sequence_features()
