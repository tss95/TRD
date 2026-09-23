"""Explicit paper profiles with no environment-dependent configuration merges."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def namespace(value: Any) -> Any:
    """Expose a plain nested mapping to the extracted scientific components."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [namespace(item) for item in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the supported single-device experiment contract."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject unsupported geometry and schedules before constructing a model."""
    if config['host'] not in {'pmt', 'ts2vec', 'patchtst'}:
        raise ValueError('Unknown host')
    if config['dataset'] not in {'circor', 'openpack'}:
        raise ValueError('Unknown dataset')
    for value in (config['channels'], config['classes'], config['data']['window_samples']):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Dimensions must be positive integers')
    span, fade = config['trd']['span_samples'], config['trd']['fade_samples']
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (span, fade)):
        raise ValueError('Span and fade must be integer sample counts')
    if not 0 <= 2 * fade < span < config['data']['window_samples']:
        raise ValueError('Require 0 <= 2*fade < span < window')
    weight = config['trd']['weight']
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('TRD weight must be finite and nonnegative')
    training = config['training']
    if not 0 < training['updates'] <= training['schedule_steps']:
        raise ValueError('Updates must lie within the declared schedule')
    if training['precision'] != 'float32':
        raise ValueError('The principal profiles require FP32')
    if config['host'] == 'pmt':
        shared = config['encoder']['pma']['shared']
        if (
            not shared['after_cnn_encoder']
            or shared['after_neighborhood_encoder']
            or shared['bidirectional']
            or not config['encoder']['use_weight_sharing']
        ):
            raise ValueError('Only the evaluated shared, unidirectional PMT configuration is supported')


def probe_config(config: dict[str, Any], data: str | Path) -> SimpleNamespace:
    """Build the small dataset/readout interface from an explicit profile."""
    return namespace(
        {
            'data_path': str(data),
            'segmentation_data': config['data'],
            'segmentation_probe': config['probe'],
            'data_parameters': {
                'num_channels': config['channels'],
                'num_classes': config['classes'],
                'seq_len_inferred': config['data']['window_samples'],
            },
        }
    )
