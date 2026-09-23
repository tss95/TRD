"""Score saved readouts on held-out recordings without fitting or selection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile

import torch

from trd.checkpoint import load_checkpoint, sha256
from trd.config import probe_config
from trd.evaluation.probe import SegmentationHead, SegmentationProbeTask
from trd.train import configure_numerics


def evaluate(
    checkpoint: Path,
    data: Path,
    heads: Path,
    output: Path,
    *,
    device: str,
    split: str = 'test',
    cache: Path | None = None,
) -> dict:
    """Require the original encoder and manifest, then apply saved heads and scaling."""
    if output.exists():
        raise FileExistsError(f'Refusing to replace results: {output}')
    if split not in {'val', 'test'}:
        raise ValueError('Evaluation requires val or test')
    files = sorted(heads.glob('*.pt'))
    if not files:
        raise ValueError('No saved readout heads')
    model, config, step = load_checkpoint(checkpoint, device)
    configure_numerics(config, device)
    task = SegmentationProbeTask(base_model=model, device=device, cfg=probe_config(config, data))
    checkpoint_hash, manifest_hash = sha256(checkpoint), sha256(data / 'manifest.json')
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='trd-eval-', dir=cache) as temporary:
        features = task._extract(split, Path(temporary))
        results = {}
        for file in files:
            state = torch.load(file, map_location='cpu', weights_only=True)
            if state['format_version'] != 1 or state['config'] != config:
                raise ValueError('Readout configuration does not match the encoder')
            if state['checkpoint_sha256'] != checkpoint_hash or state['manifest_sha256'] != manifest_hash:
                raise ValueError('Readout was fitted against another checkpoint or data manifest')
            if state['class_names'] != features.dataset.class_names:
                raise ValueError('Readout class order differs from the prepared dataset')
            head = SegmentationHead(
                state['input_dim'],
                config['classes'],
                kind=state['head_kind'],
                hidden=config['probe']['temporal_hidden_dim'],
                kernel=config['probe']['temporal_kernel_size'],
            ).to(device)
            head.load_state_dict(state['head'], strict=True)
            scored, _ = task._score(head, features, state['feature_mean'].numpy(), state['feature_scale'].numpy())
            results[file.stem] = {'metrics': scored['metrics'], 'selected_epoch': state['selected_epoch']}
    result = {
        'dataset': config['dataset'],
        'host': config['host'],
        'seed': config['seed'],
        'step': step,
        'mode': 'trd' if config['trd']['weight'] > 0 else 'native',
        'split': split,
        'checkpoint_sha256': checkpoint_hash,
        'readouts': results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main() -> None:
    """Apply frozen selected heads to one specified held-out split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    evaluate(
        args.checkpoint, args.data, args.heads, args.output, device=args.device, split=args.split, cache=args.cache
    )


if __name__ == '__main__':
    main()
