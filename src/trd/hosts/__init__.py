"""The three hosts evaluated in the principal paper experiments."""

from typing import Any
from torch import nn


def build_host(config: dict[str, Any]) -> nn.Module:
    """Construct only the host selected by the explicit configuration."""
    if config['host'] == 'ts2vec':
        from trd.hosts.ts2vec.model import TS2Vec

        return TS2Vec(config)
    if config['host'] == 'patchtst':
        from trd.hosts.patchtst.model import PatchTST

        return PatchTST(config)
    if config['host'] == 'pmt':
        from trd.hosts.pmt.model import PMT

        return PMT(config)
    raise ValueError(f"Unknown host {config['host']!r}")
