"""Raw-sample interval reversal with a linear crossfade."""

import torch


def corrupt_with_reversed_segment(
    x: torch.Tensor,
    span_samples: int,
    crossfade_samples: int,
    generator: torch.Generator | None = None,
    *,
    starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse one contiguous time segment per sample of x [B, L, C].

    Returns (corrupted [B, L, C], starts [B] long). Linear crossfade at the
    span edges reduces splice discontinuities without establishing shortcut
    immunity. Explicit starts support valid-window/padding-aware callers;
    omitting them preserves the original uniform placement and RNG draws.
    """
    if x.dim() != 3:
        raise ValueError(f"corrupt_with_reversed_segment expects [B, L, C], got {tuple(x.shape)}")
    B, L, _ = x.shape
    span = int(span_samples)
    if not 2 * crossfade_samples < span < L:
        raise ValueError(f"need 2*crossfade ({2 * crossfade_samples}) < span ({span}) < L ({L})")
    if starts is None:
        starts = torch.randint(0, L - span + 1, (B,), device=x.device, generator=generator)
    elif (
        starts.shape != (B,)
        or starts.dtype != torch.long
        or starts.device != x.device
        or not bool(((starts >= 0) & (starts + span <= L)).all())
    ):
        raise ValueError("Explicit reversal starts must be in-bounds long [B] on the input device")

    # Batched gather/scatter preserves the exact placement and crossfade math,
    # without one CUDA-to-Python synchronization for every sample's start.
    indices = starts[:, None] + torch.arange(span, device=x.device)[None, :]
    indices = indices[:, :, None].expand(-1, -1, x.size(2))
    segment = x.gather(1, indices.flip(1))
    if crossfade_samples:
        k = int(crossfade_samples)
        ramp = torch.linspace(0.0, 1.0, k, device=x.device).view(1, -1, 1)
        original = x.gather(1, indices)
        # Assign only the edges, retaining original dtype/NaN behavior in the
        # untouched reversed core (multiplication by zero would propagate NaN).
        segment[:, :k] = ramp * segment[:, :k] + (1 - ramp) * original[:, :k]
        segment[:, -k:] = ramp.flip(1) * segment[:, -k:] + (1 - ramp.flip(1)) * original[:, -k:]
    return x.clone().scatter_(1, indices, segment), starts
