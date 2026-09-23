"""Frozen linear/contextual segmentation probes on linked recording groups."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from trd.features import BackboneOutput
from trd.storage import require_disk_budget
from trd.data.windows import SegmentationDataset
from trd.evaluation.metrics import score_recordings, stitch_probabilities

logger = logging.getLogger(__name__)


def align_dense_features(
    output: BackboneOutput, *, input_length: int, grid_step: int, sample_coordinates: bool = False
) -> torch.Tensor:
    """Interpolate native token centers onto an explicit fixed sample grid.

    Coordinate-free backbones retain PMT's existing uniform-grid interpretation.
    Constant edge extension is explicit; invalid token intervals are rejected.
    """
    features = output.sequence_features()
    if features.shape[1] < 1 or input_length < 1 or grid_step < 1 or input_length % grid_step:
        raise ValueError("Invalid dense feature/grid geometry")
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite dense features")
    if output.mask is not None and not bool(output.mask.all()):
        raise ValueError("Dense feature extraction left invalid token intervals")
    positions = output.sample_positions
    if positions is None:
        positions = (torch.arange(features.shape[1], device=features.device) + 0.5) * input_length / features.shape[
            1
        ] - 0.5
    positions = positions.to(device=features.device, dtype=torch.float32)
    if (
        positions.ndim != 1
        or positions.numel() != features.shape[1]
        or not torch.isfinite(positions).all()
        or bool((positions[1:] <= positions[:-1]).any())
    ):
        raise ValueError("Native sample positions must be finite and strictly increasing")
    targets = (torch.arange(input_length // grid_step, device=features.device) + 0.5) * grid_step - 0.5
    if sample_coordinates:
        targets = torch.arange(input_length // grid_step, device=features.device) * grid_step
    targets = targets.clamp(positions[0], positions[-1])
    if positions.numel() == 1:
        return features.expand(-1, targets.numel(), -1)
    right = torch.searchsorted(positions.contiguous(), targets.contiguous()).clamp(1, len(positions) - 1)
    left = right - 1
    weight = ((targets - positions[left]) / (positions[right] - positions[left])).to(features.dtype)
    return torch.lerp(features[:, left], features[:, right], weight[None, :, None])


def label_group_order(dataset: SegmentationDataset, *, seed: int) -> list[str]:
    """Return the deterministic group order used by every nested label budget."""
    groups = sorted({record["group_id"] for record in dataset.records})
    random.Random(seed).shuffle(groups)
    return groups


class SegmentationHead(nn.Module):
    """A pointwise linear readout or two-convolution temporal readout."""

    def __init__(self, dim: int, classes: int, *, kind: str, hidden: int, kernel: int) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Conv1d(dim, classes, 1)
        elif kind == "temporal":
            self.network = nn.Sequential(
                nn.Conv1d(dim, hidden, kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Conv1d(hidden, classes, 1),
            )
        else:
            raise ValueError(f"Unknown segmentation head: {kind}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits [batch,class,time] from features [batch,feature,time]."""
        return self.network(features)


@dataclass
class _Features:
    values: np.ndarray
    labels: np.ndarray
    observed: np.ndarray
    dataset: SegmentationDataset


class _CachedWindows(Dataset):
    def __init__(self, features: _Features, indices: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> None:
        self.features, self.indices, self.mean, self.scale = features, indices, mean, scale

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        values = (np.asarray(self.features.values[row]) - self.mean) / self.scale
        return torch.from_numpy(np.array(values.T, copy=True)), torch.from_numpy(self.features.labels[row].copy())


def feature_standardization(features: _Features) -> tuple[np.ndarray, np.ndarray]:
    """Fit stable feature moments using observed TRAIN inputs only, without labels."""
    count = 0
    mean = np.zeros(features.values.shape[-1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(features.values), 64):
        values = np.asarray(features.values[start : start + 64], dtype=np.float64)[
            features.observed[start : start + 64]
        ]
        if len(values) == 0:
            continue
        batch_mean = values.mean(axis=0)
        delta = batch_mean - mean
        new_count = count + len(values)
        m2 += ((values - batch_mean) ** 2).sum(axis=0) + delta**2 * count * len(values) / new_count
        mean += delta * len(values) / new_count
        count = new_count
    if count == 0:
        raise ValueError("No observed training features")
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-5).astype(np.float32)


class SegmentationProbeTask:
    """Fit frozen heads, choose epochs on val, and score complete recording timelines."""

    def __init__(self, *, base_model: nn.Module, device: str | torch.device, cfg: Any, rank: int = 0) -> None:
        self.base_model, self.device, self.cfg, self.rank = base_model, torch.device(device), cfg, rank
        self.parameters = cfg.segmentation_probe

    def _extract(self, split: str, directory: Path) -> _Features:
        dataset = SegmentationDataset(self.cfg.data_path, split=split, parameters=self.cfg.segmentation_data, ssl=False)
        dataset.validate_model_dimensions(self.cfg.data_parameters)
        if dataset.manifest.get("scoring_protocol", "interval") != self.parameters.scoring_protocol:
            raise ValueError("Prepared data and configured segmentation scoring protocols differ")
        loader = DataLoader(dataset, batch_size=self.parameters.embedding_batch_size, shuffle=False, num_workers=0)
        backbone = self.base_model.module if hasattr(self.base_model, "module") else self.base_model
        values = None
        labels = np.empty((len(dataset), dataset.target_length), dtype=np.int64)
        observed = np.empty_like(labels, dtype=bool)
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(self.device).transpose(1, 2).contiguous()
                output = backbone.encode(signal)
                if not isinstance(output, BackboneOutput):
                    raise TypeError("Segmentation requires native BackboneOutput features")
                z = (
                    align_dense_features(
                        output,
                        input_length=dataset.length,
                        grid_step=dataset.grid_step,
                        sample_coordinates=dataset.target_coordinates == "sample",
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                if values is None:
                    shape = (len(dataset), *z.shape[1:])
                    require_disk_budget(directory, math.prod(shape) * 4)
                    values = np.lib.format.open_memmap(
                        directory / f"{split}.npy", mode="w+", dtype=np.float32, shape=shape
                    )
                indices = batch["index"].numpy()
                values[indices] = z
                labels[indices] = batch["labels"].numpy()
                observed[indices] = batch["observed"].numpy()
                if int(indices[-1]) + 1 == len(dataset) or int(indices[0]) % 512 == 0:
                    logger.info(
                        "Segmentation feature cache split=%s windows=%d/%d", split, int(indices[-1]) + 1, len(dataset)
                    )
        if values is None:
            raise ValueError(f"Empty feature extraction: {split}")
        values.flush()
        return _Features(values, labels, observed, dataset)

    def _predict(self, head: nn.Module, features: _Features, mean: np.ndarray, scale: np.ndarray) -> list[np.ndarray]:
        loader = DataLoader(
            _CachedWindows(features, np.arange(len(features.values)), mean, scale),
            batch_size=self.parameters.batch_size,
            shuffle=False,
            num_workers=0,
        )
        predictions = np.empty(
            (len(features.values), features.dataset.target_length, features.dataset.num_classes), dtype=np.float32
        )
        head.eval()
        cursor = 0
        with torch.no_grad():
            for inputs, _ in loader:
                probs = head(inputs.to(self.device)).softmax(dim=1).transpose(1, 2).cpu().numpy()
                predictions[cursor : cursor + len(probs)] = probs
                cursor += len(probs)
        return stitch_probabilities(features.dataset, predictions)

    def _score(
        self, head: nn.Module, features: _Features, mean: np.ndarray, scale: np.ndarray
    ) -> tuple[dict, list[np.ndarray]]:
        probs = self._predict(head, features, mean, scale)
        if self.parameters.scoring_protocol == "openpack_1hz":
            from trd.evaluation.openpack import score_openpack_recordings

            return score_openpack_recordings(features.dataset, probs), probs
        return (
            score_recordings(
                features.dataset,
                probs,
                event_classes=self.parameters.event_classes,
                tolerances=self.parameters.event_tolerances_seconds,
            ),
            probs,
        )

    def _fit(
        self, train: _Features, val: _Features, indices: np.ndarray, *, kind: str, mean: np.ndarray, scale: np.ndarray
    ) -> tuple[SegmentationHead, int]:
        if not np.any(train.labels[indices] >= 0):
            raise ValueError("Selected label groups contain no annotated frames")
        torch.manual_seed(self.parameters.head_seed)
        head = SegmentationHead(
            train.values.shape[-1],
            train.dataset.num_classes,
            kind=kind,
            hidden=self.parameters.temporal_hidden_dim,
            kernel=self.parameters.temporal_kernel_size,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=self.parameters.learning_rate, weight_decay=self.parameters.weight_decay
        )
        generator = torch.Generator().manual_seed(self.parameters.head_seed)
        loader = DataLoader(
            _CachedWindows(train, indices, mean, scale),
            batch_size=self.parameters.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        best_score, best_state, best_epoch = -1.0, None, 0
        for epoch in range(1, self.parameters.epochs + 1):
            head.train()
            for inputs, targets in loader:
                if not bool((targets >= 0).any()):
                    continue  # An explicitly unknown-only batch contributes no supervised loss.
                optimizer.zero_grad(set_to_none=True)
                logits = head(inputs.to(self.device))
                loss = F.cross_entropy(logits, targets.to(self.device), ignore_index=-1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite segmentation probe loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise RuntimeError("Nonfinite segmentation probe gradients")
                optimizer.step()
            result, _ = self._score(head, val, mean, scale)
            score = result["metrics"]["state_macro_f1"]
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            if epoch == 1 or epoch % 5 == 0 or epoch == self.parameters.epochs:
                logger.info("Segmentation head=%s epoch=%d val_state_f1=%.5f", kind, epoch, score)
        head.load_state_dict(best_state, strict=True)
        return head, best_epoch
