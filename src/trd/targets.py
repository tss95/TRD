"""Token targets; the patch helper preserves the paper's PMT coordinates."""

import torch

from trd.corruption import corrupt_with_reversed_segment


def sample_reversal(
    x: torch.Tensor, span: int, fade: int, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a reversed BTC view, sample membership, validity and interval starts.

    Uniform placements fit inside one contiguous valid region per window.
    NaN timestamps are invalid when no explicit mask is supplied.
    """
    if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError('Expected a nonempty floating point [B,T,C] waveform')
    if isinstance(span, bool) or not isinstance(span, int) or isinstance(fade, bool) or not isinstance(fade, int):
        raise ValueError('Span and fade must be integer sample counts')
    if not 0 <= 2 * fade < span < x.size(1) or bool(torch.isinf(x).any()):
        raise ValueError('Require 0 <= 2*fade < span < length and no infinite samples')
    if valid is None:
        valid = ~x.isnan().any(dim=-1)
    if valid.dtype != torch.bool or valid.shape != x.shape[:2] or valid.device != x.device:
        raise ValueError('Validity must be boolean [B,T] on the waveform device')
    counts = valid.sum(dim=1)
    if not bool((counts > span).all()):
        raise ValueError('Each valid region must be longer than the reversal interval')
    offsets = valid.long().argmax(dim=1)
    positions = torch.arange(x.size(1), device=x.device)[None, :]
    expected = (positions >= offsets[:, None]) & (positions < (offsets + counts)[:, None])
    if not torch.equal(valid, expected) or not bool(torch.isfinite(x[valid]).all()):
        raise ValueError('Expected one contiguous finite valid region per waveform')
    if bool(valid.all()):
        reversed_x, starts = corrupt_with_reversed_segment(x, span, fade)
    else:
        starts = offsets + (torch.rand(x.size(0), device=x.device) * (counts - span + 1)).long()
        reversed_x, starts = corrupt_with_reversed_segment(x, span, fade, starts=starts)
    labels = (positions >= starts[:, None]) & (positions < starts[:, None] + span)
    return reversed_x, labels.float(), valid, starts


def token_labels_from_spans(
    starts: torch.Tensor, span_samples: int, num_tokens: int, tokenizer_stride: int, tokenizer_patch: int
) -> torch.Tensor:
    """[B, T] float labels: 1 where >= half the token window overlaps the span."""
    tok_centers = torch.arange(num_tokens, device=starts.device) * tokenizer_stride + tokenizer_patch // 2
    lo = starts.view(-1, 1)
    hi = (starts + int(span_samples)).view(-1, 1)
    half = tokenizer_patch // 2
    ov_lo = torch.maximum(tok_centers.view(1, -1) - half, lo)
    ov_hi = torch.minimum(tok_centers.view(1, -1) + half, hi)
    overlap = (ov_hi - ov_lo).clamp(min=0).float()
    return (overlap >= 0.5 * tokenizer_patch).float()
