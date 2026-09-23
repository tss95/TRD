"""Lazy prepared-recording windows for SSL and masked temporal probes."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from trd.data.temporal import Interval, TemporalWindow, rasterize_intervals, recording_windows, validate_intervals

InputLayout = Literal["BCT", "BTC"]


class SegmentationDataset(Dataset):
    """Read normalized signals while retaining absolute recording and label geometry."""

    # Manifest format v1 fixes on-disk arrays to TC; __getitem__ emits CT windows.
    # Lazy datasets bypass normalize_dataset_spec and declare their batched output here.
    storage_layout = "TC"
    sample_layout: InputLayout = "BCT"

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        parameters: Any,
        ssl: bool,
        training: bool = False,
        single_view: bool = False,
    ) -> None:
        if single_view and not ssl:
            raise ValueError("Single-view input is an SSL-only dataset contract")
        self.single_view = bool(single_view)
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format_version") != 1 or split not in {"train", "val", "test"}:
            raise ValueError("Unsupported prepared-recording manifest or split")
        self.rate = int(self.manifest["sampling_rate_hz"])
        self.target_coordinates = self.manifest.get("target_coordinates", "midpoint")
        if self.target_coordinates not in {"midpoint", "sample"}:
            raise ValueError("Unknown prepared target coordinates")
        self.class_names = list(self.manifest["class_names"])
        self.num_classes = len(self.class_names)
        self.num_channels = len(self.manifest["channel_names"])
        self.length = int(parameters.window_samples)
        self.stride = int(parameters.stride_samples)
        target_rate = int(parameters.target_rate_hz)
        if target_rate < 1 or self.rate % target_rate:
            raise ValueError("Target rate must divide the native sampling rate")
        self.grid_step = self.rate // target_rate
        if self.length % self.grid_step or self.stride % self.grid_step:
            raise ValueError("Window length and stride must be aligned to the target grid")
        self.target_length = self.length // self.grid_step
        self.parameters = parameters
        self.ssl = bool(ssl)
        self.training = bool(training)
        all_records = self.manifest["records"]
        ids = [r["record_id"] for r in all_records]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate recording IDs in manifest")
        memberships: dict[str, set[str]] = {}
        for record in all_records:
            memberships.setdefault(record["group_id"], set()).add(record["split"])
        if any(len(values) != 1 for values in memberships.values()):
            raise ValueError("An identity group crosses dataset partitions")
        self.records = [r for r in all_records if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty prepared split: {split}")
        self.intervals = []
        self.event_intervals = []
        self.windows: list[TemporalWindow] = []
        for i, record in enumerate(self.records):
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts or not (self.root / relative).is_file():
                raise ValueError(f"Missing or unsafe prepared waveform path: {relative}")
            annotations = [Interval(float(a), float(b), int(c)) for a, b, c in record["annotations"]]
            validate_intervals(annotations, duration=record["num_samples"] / self.rate, num_classes=self.num_classes)
            self.intervals.append(annotations)
            events = [
                Interval(float(a), float(b), int(c))
                for a, b, c in record.get("event_annotations", record["annotations"])
            ]
            for event in events:
                validate_intervals([event], duration=record["num_samples"] / self.rate, num_classes=self.num_classes)
            self.event_intervals.append(events)
            for start, valid in recording_windows(
                record["num_samples"],
                length=self.length,
                stride=self.stride,
                grid_step=1 if ssl else self.grid_step,
                allow_padding=not ssl,
            ):
                self.windows.append(TemporalWindow(i, start, valid))
        self._arrays: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def validate_model_dimensions(self, parameters: Any) -> None:
        """Reject disagreement between prepared signal/classes and model metadata."""
        layout = getattr(parameters, "sample_layout", None)
        if layout is not None and layout != self.sample_layout:
            raise ValueError(
                f"Segmentation reader emits {self.sample_layout}, but data_parameters.sample_layout={layout!r}"
            )
        expected = (parameters.num_channels, parameters.num_classes, parameters.seq_len_inferred)
        observed = (self.num_channels, self.num_classes, self.length)
        if observed != expected:
            raise ValueError(f"Prepared segmentation dimensions {observed} do not match model metadata {expected}")

    def __getstate__(self) -> dict:
        """Open memmaps independently in spawned dataloader workers."""
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state

    def _waveform(self, record_index: int) -> np.ndarray:
        if record_index not in self._arrays:
            record = self.records[record_index]
            values = np.load(self.root / record["path"], mmap_mode="r", allow_pickle=False)
            if values.dtype != np.float32 or values.shape != (record["num_samples"], self.num_channels):
                raise ValueError(f"Prepared signal shape/dtype mismatch: {record['record_id']}")
            self._arrays[record_index] = values
            if len(self._arrays) > 4:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(record_index)
        return self._arrays[record_index]

    def targets(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return dense labels and independent observed-signal support on the probe grid."""
        window = self.windows[index]
        local_times = self.target_sample_positions()
        observed = local_times < window.valid_samples
        times = (window.start + local_times) / self.rate
        labels = rasterize_intervals(self.intervals[window.record_index], times)
        labels[~observed] = -1
        return labels, observed

    def target_sample_positions(self) -> np.ndarray:
        """Return target coordinates in samples under the manifest's explicit contract."""
        if self.target_coordinates == "sample":
            return np.arange(self.target_length) * self.grid_step
        return (np.arange(self.target_length) + 0.5) * self.grid_step

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor | int]:
        window = self.windows[index]
        array = self._waveform(window.record_index)[window.start : window.start + window.valid_samples]
        signal = torch.zeros((self.num_channels, self.length), dtype=torch.float32)
        signal[:, : window.valid_samples] = torch.from_numpy(np.array(array.T, copy=True))
        if not torch.isfinite(signal).all():
            raise ValueError("Nonfinite prepared signal")
        if self.ssl:
            # Labels are never consulted for SSL crop eligibility or corruption.
            if self.single_view:
                return signal, torch.tensor(index)
            weak, strong = signal.clone(), signal.clone()
            if self.training:
                if self.parameters.weak_noise_std > 0:
                    weak += torch.randn_like(weak) * self.parameters.weak_noise_std
                if self.parameters.strong_noise_std > 0:
                    strong += torch.randn_like(strong) * self.parameters.strong_noise_std
            return signal, torch.tensor(0), weak, strong, torch.tensor(False), torch.tensor(index)
        labels, observed = self.targets(index)
        return {
            "signal": signal,
            "labels": torch.from_numpy(labels),
            "observed": torch.from_numpy(observed),
            "index": index,
            "record_index": window.record_index,
            "start_sample": window.start,
        }
