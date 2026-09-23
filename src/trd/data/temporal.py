"""Recording-time interval targets and deterministic windows for dense probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Interval:
    """Half-open annotation in seconds; class -1 denotes unknown coverage."""

    start: float
    end: float
    label: int


@dataclass(frozen=True)
class TemporalWindow:
    """Window coordinates in original recording samples, before right padding."""

    record_index: int
    start: int
    valid_samples: int


def validate_intervals(intervals: Sequence[Interval], *, duration: float, num_classes: int) -> None:
    """Reject invalid classes, overlaps and out-of-record annotations; allow gaps."""
    if not math.isfinite(duration) or duration <= 0 or num_classes < 1:
        raise ValueError("Interval validation requires a positive duration and class count")
    previous_end = 0.0
    for interval in intervals:
        if not (math.isfinite(interval.start) and math.isfinite(interval.end)):
            raise ValueError("Interval endpoints must be finite")
        if interval.start < 0 or interval.end <= interval.start or interval.end > duration + 1e-6:
            raise ValueError(f"Invalid interval bounds: {interval}, duration={duration}")
        if interval.start < previous_end - 1e-6:
            raise ValueError(f"Overlapping or unordered intervals: {interval}")
        if not isinstance(interval.label, int) or not -1 <= interval.label < num_classes:
            raise ValueError(f"Invalid interval class: {interval.label}")
        previous_end = interval.end


def rasterize_intervals(intervals: Sequence[Interval], times: np.ndarray) -> np.ndarray:
    """Assign labels at explicit times without filling annotation gaps."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Target times must be finite, increasing and one-dimensional")
    labels = np.full(times.shape, -1, dtype=np.int64)
    for interval in intervals:
        left = np.searchsorted(times, interval.start, side="left")
        right = np.searchsorted(times, interval.end, side="left")
        labels[left:right] = interval.label
    return labels


def recording_windows(
    num_samples: int, *, length: int, stride: int, grid_step: int = 1, allow_padding: bool = True
) -> list[tuple[int, int]]:
    """Cover a recording with full windows and one grid-aligned tail window.

    Returns (start, valid_samples). With padding disabled every window is real
    signal, including a final window ending exactly at the recording endpoint.
    """
    if min(num_samples, length, stride, grid_step) < 1 or stride > length:
        raise ValueError("Window sizes must be positive and stride cannot exceed length")
    if length % grid_step or stride % grid_step:
        raise ValueError("Window length and stride must align with the target grid")
    if num_samples < length:
        if not allow_padding:
            raise ValueError("SSL requires a full observed window; recording is shorter than the configured context")
        return [(0, num_samples)]
    starts = list(range(0, num_samples - length + 1, stride))
    tail = num_samples - length
    if allow_padding:
        tail = math.ceil(tail / grid_step) * grid_step
    if starts[-1] != tail:
        starts.append(tail)
    return [(start, min(length, num_samples - start)) for start in starts]


def contiguous_events(labels: np.ndarray, *, label: int, frame_seconds: float) -> list[tuple[float, float]]:
    """Convert a discrete state track into all intervals of one class."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or frame_seconds <= 0:
        raise ValueError("Events require a one-dimensional state track and positive frame duration")
    selected = np.r_[False, labels == label, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(selected) == 1)
    ends = np.flatnonzero(np.diff(selected) == -1)
    return [(float(a * frame_seconds), float(b * frame_seconds)) for a, b in zip(starts, ends)]


def match_event_onsets(
    reference: Sequence[float], prediction: Sequence[float], *, tolerance_seconds: float
) -> tuple[int, int, int, list[float]]:
    """Match sorted onsets one-to-one with maximum cardinality within tolerance.

    Earliest feasible matching has maximum cardinality for ordered scalar times
    with a shared tolerance. Timing errors use that deterministic matching; this
    does not claim minimum total timing error among all optimal matchings.
    """
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    prediction = np.sort(np.asarray(prediction, dtype=np.float64))
    if (
        reference.ndim != 1
        or prediction.ndim != 1
        or not np.isfinite(reference).all()
        or not np.isfinite(prediction).all()
        or np.any(reference < 0)
        or np.any(prediction < 0)
        or not math.isfinite(tolerance_seconds)
        or tolerance_seconds < 0
    ):
        raise ValueError("Onsets and tolerance must be finite, one-dimensional and nonnegative")
    i = j = 0
    errors: list[float] = []
    while i < len(reference) and j < len(prediction):
        delta = float(prediction[j] - reference[i])
        if abs(delta) <= tolerance_seconds + 1e-12:
            errors.append(abs(delta))
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    tp = len(errors)
    return tp, len(prediction) - tp, len(reference) - tp, errors
