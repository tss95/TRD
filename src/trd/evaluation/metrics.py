"""Recording-level state and one-to-one event metrics for interval annotations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from trd.data.windows import SegmentationDataset
from trd.data.temporal import contiguous_events, match_event_onsets, rasterize_intervals


def stitch_probabilities(dataset: SegmentationDataset, windows: np.ndarray) -> list[np.ndarray]:
    """Average overlapping predictions on absolute recording grids, excluding padding."""
    expected = (len(dataset), dataset.target_length, dataset.num_classes)
    if windows.shape != expected or not np.isfinite(windows).all():
        raise ValueError(f"Expected finite window probabilities {expected}, got {windows.shape}")
    totals = [
        np.zeros((math.ceil(r["num_samples"] / dataset.grid_step), dataset.num_classes), dtype=np.float64)
        for r in dataset.records
    ]
    counts = [np.zeros(len(values), dtype=np.int64) for values in totals]
    for i, window in enumerate(dataset.windows):
        _, observed = dataset.targets(i)
        positions = window.start // dataset.grid_step + np.flatnonzero(observed)
        totals[window.record_index][positions] += windows[i, observed]
        counts[window.record_index][positions] += 1
    for i, record in enumerate(dataset.records):
        offset = 0.0 if dataset.target_coordinates == "sample" else 0.5
        observed = (np.arange(len(totals[i])) + offset) * dataset.grid_step < record["num_samples"]
        if np.any(counts[i][observed] == 0):
            raise ValueError(f"Uncovered predictions in {record['record_id']}")
        totals[i] /= np.maximum(counts[i], 1)[:, None]
    return totals


def _confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    tp = confusion.diagonal().astype(float)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    count = int(confusion.sum())
    if count == 0:
        raise ValueError("No annotated frames available for scoring")
    return {"state_macro_f1": float(f1.mean()), "state_accuracy": float(tp.sum() / count), "labeled_frames": count}


def _event_summary(counts: tuple[int, int, int, list[float]]) -> dict[str, float | int | None]:
    tp, fp, fn, errors = counts
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "matched_mae_seconds": float(np.mean(errors)) if errors else None,
    }


def score_recordings(
    dataset: SegmentationDataset,
    probabilities: Sequence[np.ndarray],
    *,
    event_classes: Sequence[int],
    tolerances: Sequence[float],
) -> dict:
    """Score real annotation coverage and events away from censored annotation edges.

    Event scoring excludes a max(tolerances) margin at both ends of each known
    annotation run, for both reference and prediction. This avoids manufacturing
    event onsets at unknown-to-known transitions. Ground-truth event times remain
    the original continuous-time annotation starts, not rasterized timestamps.
    """
    if len(probabilities) != len(dataset.records):
        raise ValueError("Prediction and recording counts differ")
    if not tolerances or any(not np.isfinite(t) or t <= 0 for t in tolerances):
        raise ValueError("Positive finite event tolerances are required")
    confusion = np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int64)
    grouped: dict[str, np.ndarray] = {}
    event_counts = {float(t): {str(c): [0, 0, 0, []] for c in [*event_classes, "untyped"]} for t in tolerances}
    frame_seconds = dataset.grid_step / dataset.rate
    margin = max(tolerances)
    for i, (record, probs) in enumerate(zip(dataset.records, probabilities)):
        n = math.ceil(record["num_samples"] / dataset.grid_step)
        if probs.shape != (n, dataset.num_classes) or not np.isfinite(probs).all():
            raise ValueError(f"Invalid recording probabilities: {record['record_id']}")
        times = (np.arange(n) + 0.5) * frame_seconds
        truth = rasterize_intervals(dataset.intervals[i], times)
        truth[times >= record["num_samples"] / dataset.rate] = -1
        valid = truth >= 0
        prediction = probs.argmax(axis=1)
        local_confusion = np.bincount(
            dataset.num_classes * truth[valid] + prediction[valid], minlength=dataset.num_classes**2
        ).reshape(dataset.num_classes, dataset.num_classes)
        confusion += local_confusion
        grouped.setdefault(record["group_id"], np.zeros_like(confusion))[:] += local_confusion
        prediction = np.where(valid, prediction, -1)
        support = contiguous_events(valid.astype(np.int64), label=1, frame_seconds=frame_seconds)

        def eligible(onset: float) -> bool:
            return any(left + margin < onset < right - margin for left, right in support)

        references = {
            str(c): [a.start for a in dataset.event_intervals[i] if a.label == c and eligible(a.start)]
            for c in event_classes
        }
        predictions = {
            str(c): [a for a, _ in contiguous_events(prediction, label=c, frame_seconds=frame_seconds) if eligible(a)]
            for c in event_classes
        }
        references["untyped"] = sorted(a for values in references.values() for a in values)
        sound_track = np.isin(prediction, event_classes).astype(np.int64)
        predictions["untyped"] = [
            a for a, _ in contiguous_events(sound_track, label=1, frame_seconds=frame_seconds) if eligible(a)
        ]
        for tolerance in tolerances:
            for kind in references:
                tp, fp, fn, errors = match_event_onsets(
                    references[kind], predictions[kind], tolerance_seconds=tolerance
                )
                accumulated = event_counts[float(tolerance)][kind]
                accumulated[0] += tp
                accumulated[1] += fp
                accumulated[2] += fn
                accumulated[3].extend(errors)
    metrics = _confusion_metrics(confusion)
    groups = {key: _confusion_metrics(value) for key, value in grouped.items() if value.sum() > 0}
    metrics["group_mean_state_macro_f1"] = float(np.mean([value["state_macro_f1"] for value in groups.values()]))
    metrics["evaluated_groups"] = len(groups)
    events = {}
    for tolerance, classes in event_counts.items():
        tag = f"{tolerance:g}s"
        events[tag] = {kind: _event_summary(tuple(values)) for kind, values in classes.items()}
        metrics[f"untyped_event_f1_at_{tag}"] = events[tag]["untyped"]["f1"]
        metrics[f"typed_event_macro_f1_at_{tag}"] = (
            float(np.mean([events[tag][str(c)]["f1"] for c in event_classes])) if event_classes else 0.0
        )
    return {
        "metrics": metrics,
        "confusion": confusion.tolist(),
        "groups": groups,
        "events": events,
        "event_boundary_margin_seconds": margin,
    }
