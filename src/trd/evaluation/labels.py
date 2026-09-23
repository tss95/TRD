"""Deterministic labeled-time budgets independent of the waveform and its labels."""

from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np

from trd.data.windows import SegmentationDataset
from trd.data.temporal import rasterize_intervals


def labeled_time_budget(
    dataset: SegmentationDataset, *, minutes: float, block_seconds: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Select nested time blocks per linked person, then mask all other supervision.

    The permutation depends only on identities, recording lengths and label seed.
    A larger budget consumes the same prefix, including a partial final block.
    Requested minutes count selected signal; unknown labels remain unknown.
    """
    frames_float = minutes * 60 * dataset.rate / dataset.grid_step
    block_float = block_seconds * dataset.rate / dataset.grid_step
    if (
        seed < 0
        or not math.isfinite(frames_float)
        or frames_float < 1
        or not math.isfinite(block_float)
        or block_float < 1
        or not math.isclose(frames_float, round(frames_float), rel_tol=0, abs_tol=1e-7)
        or not math.isclose(block_float, round(block_float), rel_tol=0, abs_tol=1e-7)
    ):
        raise ValueError("Label budgets must be positive and aligned to the probe grid")
    wanted, block = round(frames_float), round(block_float)
    groups = sorted({r["group_id"] for r in dataset.records})
    selected = [np.zeros(math.ceil(r["num_samples"] / dataset.grid_step), dtype=bool) for r in dataset.records]
    blocks = {}
    for group in groups:
        choices = []
        for i, record in enumerate(dataset.records):
            if record["group_id"] != group:
                continue
            offset = 0.0 if dataset.target_coordinates == "sample" else 0.5
            observed_count = int(
                np.count_nonzero((np.arange(len(selected[i])) + offset) * dataset.grid_step < record["num_samples"])
            )
            choices.extend((i, start, min(start + block, observed_count)) for start in range(0, observed_count, block))
        if sum(end - start for _, start, end in choices) < wanted:
            raise ValueError(f"Insufficient observed signal for requested label budget in person {group}")
        random.Random(f"{seed}:{group}").shuffle(choices)
        remaining = wanted
        for i, start, end in choices:
            if not remaining:
                break
            end = min(end, start + remaining)
            selected[i][start:end] = True
            blocks.setdefault(dataset.records[i]["record_id"], []).append(
                [start * dataset.grid_step / dataset.rate, end * dataset.grid_step / dataset.rate]
            )
            remaining -= end - start
    labels = np.full((len(dataset), dataset.target_length), -1, dtype=np.int64)
    for i, window in enumerate(dataset.windows):
        original, observed = dataset.targets(i)
        local = np.flatnonzero(observed)
        positions = window.start // dataset.grid_step + local
        keep = local[selected[window.record_index][positions]]
        labels[i, keep] = original[keep]
    class_frames = np.zeros(dataset.num_classes, dtype=np.int64)
    group_frames = dict.fromkeys(groups, 0)
    for i, record in enumerate(dataset.records):
        offset = 0.0 if dataset.target_coordinates == "sample" else 0.5
        times = (np.arange(len(selected[i])) + offset) * dataset.grid_step / dataset.rate
        truth = rasterize_intervals(dataset.intervals[i], times)
        valid = selected[i] & (truth >= 0)
        class_frames += np.bincount(truth[valid], minlength=dataset.num_classes)
        group_frames[record["group_id"]] += int(valid.sum())
    metadata = {
        "selected_groups": groups,
        "minutes_per_group": minutes,
        "selected_signal_seconds": len(groups) * wanted * dataset.grid_step / dataset.rate,
        "labeled_seconds": int(class_frames.sum()) * dataset.grid_step / dataset.rate,
        "class_seconds": (class_frames * dataset.grid_step / dataset.rate).tolist(),
        "group_labeled_seconds": {g: n * dataset.grid_step / dataset.rate for g, n in group_frames.items()},
        "labeled_groups": sum(n > 0 for n in group_frames.values()),
        "labeled_classes": int(np.count_nonzero(class_frames)),
        "selected_time_blocks": blocks,
        "label_selection_sha256": hashlib.sha256(json.dumps(blocks, sort_keys=True).encode()).hexdigest(),
    }
    return labels, metadata
