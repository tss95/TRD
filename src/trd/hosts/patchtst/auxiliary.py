"""Local reversal and view-separated normalization for PatchTST SSL."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from trd.loss import TemporalReversalLoss
from trd.corruption import corrupt_with_reversed_segment
from trd.hosts.patchtst.vendor.patch_mask import create_patch


class ViewBatchNorm1d(nn.BatchNorm1d):
    """Share affine parameters while keeping reconstruction/TRD batch statistics apart.

    Only reconstruction updates inference buffers. The auxiliary group uses
    its own training statistics, matching a separate forward with buffer updates
    disabled. Evaluation and ordinary single-view forwards retain native BN.
    Runtime context is not checkpoint state.
    """

    view_batch_size: int | None = None
    update_running_stats: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each training view independently with shared affine weights."""
        if not self.training:
            return super().forward(x)
        if self.view_batch_size is not None:
            if x.size(0) != 2 * self.view_batch_size:
                raise ValueError("Fused PatchTST BatchNorm expects two equal view groups")
            first, second = x.split(self.view_batch_size, dim=0)
            return torch.cat((super().forward(first), self._auxiliary_forward(second)), dim=0)
        if not self.update_running_stats:
            return self._auxiliary_forward(x)
        return super().forward(x)

    def _auxiliary_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        return F.batch_norm(x, None, None, self.weight, self.bias, True, 0.0, self.eps)


def install_view_batchnorm(module: nn.Module) -> None:
    """Replace native BN modules without changing parameters or state-dict keys."""
    for name, child in list(module.named_children()):
        if type(child) is nn.BatchNorm1d:
            replacement = ViewBatchNorm1d(
                child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats
            )
            # Preserve actual parameter/buffer objects, including their dtype/device.
            replacement.weight, replacement.bias = child.weight, child.bias
            replacement.running_mean, replacement.running_var = child.running_mean, child.running_var
            replacement.num_batches_tracked = child.num_batches_tracked
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            install_view_batchnorm(child)


@contextmanager
def batchnorm_view_context(
    module: nn.Module, *, view_batch_size: int | None = None, update_running_stats: bool = True
) -> Iterator[None]:
    """Apply one forward's view grouping and restore it even when encoding fails."""
    layers = [child for child in module.modules() if isinstance(child, ViewBatchNorm1d)]
    previous = [(child.view_batch_size, child.update_running_stats) for child in layers]
    try:
        for child in layers:
            child.view_batch_size = view_batch_size
            child.update_running_stats = update_running_stats
        yield
    finally:
        for child, (size, update) in zip(layers, previous):
            child.view_batch_size, child.update_running_stats = size, update


class PatchTSTTRD(nn.Module):
    """Build raw-time reversal targets on PatchTST's actual retained patch grid."""

    def __init__(self, cfg: Any, *, embedding_dim: int) -> None:
        super().__init__()
        self.patch_length = int(cfg.model.patchtst_ssl.patch_length)
        self.stride = int(cfg.model.patchtst_ssl.stride)
        reversal = cfg.contrast_parameters.temporal_reversal
        if str(reversal.mode) != "segment":
            raise ValueError("PatchTST TRD supports segment mode only")
        if tuple(getattr(reversal, "span_tokens_choices", ())) or float(getattr(reversal, "boundary_weight", 0)) != 0:
            raise ValueError("PatchTST TRD requires the scalar membership objective")
        for name in ("span_tokens", "crossfade_samples"):
            value = getattr(reversal, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"PatchTST TRD {name} must be an integer")
        self.span_samples = int(reversal.span_tokens) * self.stride
        self.crossfade_samples = int(reversal.crossfade_samples)
        if not 0 <= 2 * self.crossfade_samples < self.span_samples < int(cfg.model.patchtst_ssl.context_length):
            raise ValueError("PatchTST TRD requires 0 <= 2*crossfade < span < context")
        # Adding an auxiliary head must not perturb common encoder initialization RNG.
        with torch.random.fork_rng(devices=[]):
            self.loss = TemporalReversalLoss(embedding_dim)

    def prepare(
        self, normalized: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reverse only valid waveform intervals; return patches, labels, mask, starts."""
        batch, length, _ = normalized.shape
        positions = torch.arange(length, device=normalized.device)[None, :]
        counts = valid.sum(dim=1)
        if not bool((counts > self.span_samples).all()):
            raise ValueError("Each PatchTST TRD input needs more valid samples than the reversal span")
        offsets = valid.to(torch.int64).argmax(dim=1)
        expected = (positions >= offsets[:, None]) & (positions < (offsets + counts)[:, None])
        if not torch.equal(valid, expected):
            raise ValueError("PatchTST TRD requires a contiguous valid interval per input")
        if bool(valid.all()):
            reversed_input, starts = corrupt_with_reversed_segment(
                normalized, self.span_samples, self.crossfade_samples
            )
        else:
            starts = offsets + (torch.rand(batch, device=normalized.device) * (counts - self.span_samples + 1)).long()
            reversed_input, starts = corrupt_with_reversed_segment(
                normalized, self.span_samples, self.crossfade_samples, starts=starts
            )
        reversed_patches, _ = create_patch(reversed_input, self.patch_length, self.stride)
        valid_patches, _ = create_patch(valid[:, :, None], self.patch_length, self.stride)
        membership = valid & (positions >= starts[:, None]) & (positions < starts[:, None] + self.span_samples)
        membership_patches, _ = create_patch(membership[:, :, None], self.patch_length, self.stride)
        valid_counts = valid_patches[:, :, 0].sum(dim=-1)
        labels = (2 * membership_patches[:, :, 0].sum(dim=-1) >= valid_counts).float()
        keep = valid_counts > 0
        positive = (labels * keep).sum(dim=1)
        if not bool(((positive > 0) & (positive < keep.sum(dim=1))).all()):
            raise ValueError("PatchTST TRD patch grid must retain positive and negative labels for every input")
        return reversed_patches, labels, keep, starts

    def compute_loss(self, encoded: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute mean membership BCE over valid patches, with channel concatenation."""
        tokens = encoded.permute(0, 3, 1, 2).flatten(2)
        if tokens.shape[:2] != labels.shape or valid.shape != labels.shape:
            raise ValueError("PatchTST TRD features, labels and validity must share [B,P]")
        # A pointwise head/BCE permits flattening without changing loss weighting.
        return self.loss.compute_loss(tokens[valid].unsqueeze(0), labels[valid].unsqueeze(0))
