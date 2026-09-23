"""Explicit time-series tensor layouts."""

import torch


def convert_input_layout(x: torch.Tensor, *, source: str, target: str) -> torch.Tensor:
    """Convert between batch/channel/time and batch/time/channel layouts."""
    if x.ndim != 3 or source not in {'BCT', 'BTC'} or target not in {'BCT', 'BTC'}:
        raise ValueError('Expected a 3D tensor and explicit BCT/BTC layouts')
    return x if source == target else x.transpose(1, 2).contiguous()
