"""Explicit random-state handling for paired training and frozen readouts."""

from collections.abc import Iterator
from contextlib import contextmanager
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and Torch random streams."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def preserve_rng() -> Iterator[None]:
    """Restore Python, NumPy, CPU and initialized CUDA random states on exit."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
