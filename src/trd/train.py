"""Pretrain one native or native+TRD encoder from an explicit paper profile."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from trd.checkpoint import save_checkpoint
from trd.config import load_config, namespace
from trd.data.windows import SegmentationDataset
from trd.hosts import build_host
from trd.random import seed_all

logger = logging.getLogger(__name__)


def configure_numerics(config: dict, device: str) -> None:
    """Set the declared FP32/TF32 numerical backend explicitly."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; use --device cpu for small checks')
    torch.backends.cuda.matmul.allow_tf32 = bool(config['training']['tf32'])
    torch.backends.cudnn.allow_tf32 = bool(config['training']['tf32'])
    torch.backends.cudnn.benchmark = device.startswith('cuda')


def train(config: dict, data: Path, output: Path, *, device: str, updates: int | None = None) -> None:
    """Run a fresh single-device training job, retaining the full schedule horizon."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    total = config['training']['updates'] if updates is None else updates
    if not 0 < total <= config['training']['schedule_steps']:
        raise ValueError('Requested updates must be positive and within the schedule')
    configure_numerics(config, device)
    seed_all(config['seed'])
    model = build_host(config).to(device).train()
    dataset = SegmentationDataset(
        data,
        split='train',
        parameters=namespace(config['data']),
        ssl=True,
        training=True,
        single_view=config['host'] != 'pmt',
    )
    if dataset.num_channels != config['channels']:
        raise ValueError('Prepared channels disagree with the selected profile')
    opt = config['optimizer']
    loader = DataLoader(
        dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=config['training']['workers'], drop_last=False
    )
    if len(loader) == 0:
        raise ValueError('Training split must contain at least one complete batch')
    if config['host'] == 'patchtst':
        from trd.hosts.patchtst.standardize import fit_training_standardizer

        fit_training_standardizer(model, loader)
        model.encoder.selected_peak_lr.fill_(config['encoder']['fixed_peak_lr'])
    groups = (
        model.optimizer_parameter_groups(namespace(opt))
        if hasattr(model, 'optimizer_parameter_groups')
        else [{'params': [p for p in model.parameters() if p.requires_grad]}]
    )
    optimizer_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[opt['optimizer_type']]
    optimizer = optimizer_cls(
        groups,
        lr=opt['max_lr'],
        betas=tuple(opt['betas']),
        weight_decay=opt['weight_decay'],
        foreach=opt['foreach'],
        fused=device.startswith('cuda') if opt['fused'] is None else opt['fused'],
    )
    tick = config['training']['schedule_tick_steps']
    scheduler = None
    if config['host'] == 'patchtst':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['encoder']['fixed_peak_lr'],
            total_steps=config['training']['schedule_steps'],
            pct_start=config['encoder']['pct_start'],
        )
    elif config['host'] == 'pmt':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['schedule_steps'] // tick, eta_min=opt['min_lr']
        )
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'run.json').write_text(
        json.dumps(
            {
                'requested_updates': total,
                'torch': str(torch.__version__),
                'device': device,
                'training_windows': len(dataset),
            },
            indent=2,
        )
        + '\n'
    )
    save_checkpoint(model, config, 0, output / 'step_0.pt')
    iterator = iter(loader)
    with (output / 'training.jsonl').open('w') as log:
        for step in range(1, total + 1):
            if config['host'] == 'pmt' and (step - 1) % tick == 0 and config['training']['advance_epoch_schedules']:
                model.set_epoch((step - 1) // tick)
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            first, second = (batch[2], batch[3]) if config['host'] == 'pmt' else (batch[0], None)
            first = first.to(device).transpose(1, 2).contiguous()
            second = None if second is None else second.to(device).transpose(1, 2).contiguous()
            optimizer.zero_grad(set_to_none=True)
            losses = model(first, second)
            loss = losses['total_loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            if hasattr(model, 'gradient_clip_parameters'):
                gates, base = model.gradient_clip_parameters()
                torch.nn.utils.clip_grad_norm_(gates, opt['gate_clip_norm'], error_if_nonfinite=True)
            else:
                base = model.parameters()
            torch.nn.utils.clip_grad_norm_(base, opt['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            if hasattr(model, 'after_optimizer_step'):
                model.after_optimizer_step()
            if scheduler is not None and (config['host'] == 'patchtst' or step % tick == 0):
                scheduler.step()
            if step == 1 or step % 100 == 0 or step == total:
                values = {'step': step, **{key: float(value.detach()) for key, value in losses.items()}}
                log.write(json.dumps(values, allow_nan=False) + '\n')
                log.flush()
                logger.info('step=%d total_loss=%.6f', step, values['total_loss'])
            if step in config['training']['checkpoint_steps'] or step == total:
                save_checkpoint(model, config, step, output / f'step_{step}.pt')


def main() -> None:
    """Run one explicitly requested experiment; no scheduler or tracking service is used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['native', 'trd'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--updates', type=int, help='Stop early, preserving the full learning-rate schedule')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    config['seed'] = args.seed
    if args.mode == 'native':
        config['trd']['weight'] = 0.0
    train(config, args.data, args.output, device=args.device, updates=args.updates)


if __name__ == '__main__':
    main()
