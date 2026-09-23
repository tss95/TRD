"""Portable tensor checkpoints, with explicit experiment metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from trd.config import validate_config
from trd.hosts import build_host


def sha256(path: str | Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(model: torch.nn.Module, config: dict[str, Any], step: int, path: Path) -> None:
    """Atomically save a clean inference checkpoint; training resume is not claimed."""
    temporary = path.with_suffix('.tmp')
    torch.save(
        {
            'format_version': 1,
            'config': config,
            'step': step,
            'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str | torch.device) -> tuple[torch.nn.Module, dict[str, Any], int]:
    """Load only tensors/basic metadata and require exact state-dict compatibility."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format_version'] != 1:
        raise ValueError('Unsupported checkpoint format')
    config = state['config']
    validate_config(config)
    model = build_host(config)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    for module in model.modules():
        if hasattr(module, 'compile_per_window'):
            module.compile_per_window = False
    return model, config, state['step']
