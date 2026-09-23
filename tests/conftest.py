"""Small synthetic experiment fixtures; no research data is read."""

import json
from pathlib import Path

import numpy as np
import torch

from trd.config import load_config

torch.set_num_threads(1)


def tiny_config(host: str, channels: int = 1) -> dict:
    """Shrink each encoder while retaining its real numerical path."""
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / f'configs/circor/{host}.yaml')
    config['channels'] = channels
    config['data'].update(window_samples=64, stride_samples=64, target_rate_hz=50)
    config['trd'].update(span_samples=16, fade_samples=2)
    config['optimizer']['batch_size'] = 2
    config['training'].update(updates=2, checkpoint_steps=[0, 2])
    config['probe'].update(
        epochs=1,
        batch_size=4,
        embedding_batch_size=4,
        label_proportions=[0.5, 1.0],
        temporal_hidden_dim=4,
        temporal_kernel_size=3,
    )
    enc = config['encoder']
    if host == 'ts2vec':
        enc.update(output_dims=16, hidden_dims=8, depth=1)
    elif host == 'patchtst':
        enc.update(
            context_length=64, patch_length=8, stride=8, d_model=16, n_heads=4, n_layers=1, d_ff=32, tile_step=32
        )
    else:
        enc.update(
            d_model=16,
            nhead=4,
            num_encoder_blocks=1,
            tokenizer_num_kernels=8,
            tokenizer_patch_size=9,
            tokenizer_stride=4,
            transformer_neighborhood_size=4,
            proj_hidden_dim=16,
            proj_output_dim=16,
        )
        enc['pma']['shared'].update(num_blocks=1, window_size=8, window_stride=8, window_checkpointing=False)
        enc['pma']['v5_4']['compile_per_window'] = False
        config['native']['contrast']['pma_contrastive_loss'].update(n_negative_samples=4, chunk_size=32)
    return config


def prepared_data(root: Path, channels: int = 1) -> Path:
    """Write labeled sine waves with separate train, validation and test groups."""
    root.mkdir()
    records = []
    for i, split in enumerate(['train', 'train', 'train', 'val', 'test']):
        name = f'record_{i}.npy'
        values = np.sin(np.arange(150, dtype=np.float32)[:, None] / 7) + np.arange(channels, dtype=np.float32)
        np.save(root / name, values)
        records.append(
            {
                'record_id': str(i),
                'group_id': str(i),
                'split': split,
                'path': name,
                'num_samples': 150,
                'labeled_seconds': 1.3,
                'annotations': [[0, 0.2, -1], [0.2, 0.5, 0], [0.5, 0.8, 1], [0.8, 1.1, 2], [1.1, 1.5, 3]],
            }
        )
    (root / 'manifest.json').write_text(
        json.dumps(
            {
                'format_version': 1,
                'sampling_rate_hz': 100,
                'channel_names': [str(i) for i in range(channels)],
                'class_names': ['S1', 'systole', 'S2', 'diastole'],
                'records': records,
                'protocol': 'synthetic',
            }
        )
    )
    return root
