"""Fit frozen readouts using training labels and validation-selected epochs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from trd.checkpoint import load_checkpoint, sha256
from trd.config import probe_config
from trd.evaluation.labels import labeled_time_budget
from trd.evaluation.probe import SegmentationProbeTask, _Features, feature_standardization, label_group_order
from trd.train import configure_numerics


def fit_heads(checkpoint: Path, data: Path, output: Path, *, device: str, cache: Path | None = None) -> None:
    """Fit both budgets and readouts without accessing the test split."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite fitted heads: {output}')
    output.mkdir(parents=True, exist_ok=True)
    model, config, step = load_checkpoint(checkpoint, device)
    configure_numerics(config, device)
    task = SegmentationProbeTask(base_model=model, device=device, cfg=probe_config(config, data))
    digest = sha256(checkpoint)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='trd-features-', dir=cache) as temporary:
        train, val = task._extract('train', Path(temporary)), task._extract('val', Path(temporary))
        mean, scale = feature_standardization(train)
        order = label_group_order(train.dataset, seed=config['probe']['label_seed'])
        budgets = [(f'lp{p:g}', p, False) for p in config['probe']['label_proportions']]
        budgets += [(f'min{m:g}', m, True) for m in config['probe']['label_minutes_per_group']]
        results = {}
        for tag, amount, is_time in budgets:
            selected_train = train
            if is_time:
                labels, selection = labeled_time_budget(
                    train.dataset,
                    minutes=amount,
                    block_seconds=config['probe']['label_block_seconds'],
                    seed=config['probe']['label_seed'],
                )
                selected_train = _Features(train.values, labels, train.observed, train.dataset)
                indices = np.flatnonzero(np.any(labels >= 0, axis=1))
            else:
                selected = order[: max(1, math.ceil(len(order) * amount))]
                selected_set = set(selected)
                indices = np.array(
                    [
                        i
                        for i, window in enumerate(train.dataset.windows)
                        if train.dataset.records[window.record_index]['group_id'] in selected_set
                    ],
                    dtype=np.int64,
                )
                selection = {'selected_groups': selected}
            for kind in config['probe']['heads']:
                head, epoch = task._fit(selected_train, val, indices, kind=kind, mean=mean, scale=scale)
                scored, _ = task._score(head, val, mean, scale)
                name = f'{kind}_{tag}'
                metadata = {
                    'format_version': 1,
                    'checkpoint_sha256': digest,
                    'encoder_step': step,
                    'config': config,
                    'head_kind': kind,
                    'input_dim': train.values.shape[-1],
                    'class_names': train.dataset.class_names,
                    'selected_epoch': epoch,
                    'selection': selection,
                    'manifest_sha256': sha256(data / 'manifest.json'),
                }
                torch.save(
                    {
                        **metadata,
                        'head': {key: value.detach().cpu() for key, value in head.state_dict().items()},
                        'feature_mean': torch.from_numpy(mean),
                        'feature_scale': torch.from_numpy(scale),
                    },
                    output / f'{name}.pt',
                )
                results[name] = {'metrics': scored['metrics'], 'selected_epoch': epoch, 'selection': selection}
        (output / 'validation.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


def main() -> None:
    """Fit the profile's frozen readouts for one trained encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, help='Temporary feature-bank directory; allow several GB')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    fit_heads(args.checkpoint, args.data, args.output, device=args.device, cache=args.cache)


if __name__ == '__main__':
    main()
