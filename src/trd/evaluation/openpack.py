"""OpenPack 1 Hz operation scores with observed-support and upstream-fill results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trd.data.windows import SegmentationDataset
from trd.evaluation.metrics import _confusion_metrics


def last_prediction_per_second(timestamps: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last class in each wall-clock second, matching the pinned toolkit."""
    timestamps, predictions = np.asarray(timestamps), np.asarray(predictions)
    if (
        timestamps.ndim != 1
        or timestamps.dtype.kind not in "iu"
        or not len(timestamps)
        or predictions.shape != timestamps.shape
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Expected matching predictions and strictly increasing integer timestamps")
    seconds = timestamps - timestamps % 1000
    keep = np.r_[seconds[1:] != seconds[:-1], True]
    return seconds[keep], predictions[keep]


def align_operation_predictions(
    reference_times: np.ndarray, prediction_times: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to annotation coverage and forward-fill; an unsupported leading label is -1.

    This reproduces the upstream alignment helper for contiguous 1 Hz targets.
    The observed mask separately identifies labels that required no filling.
    """
    reference_times = np.asarray(reference_times)
    prediction_times, prediction = np.asarray(prediction_times), np.asarray(prediction)
    if (
        reference_times.ndim != 1
        or not len(reference_times)
        or reference_times.dtype.kind not in "iu"
        or np.any(np.diff(reference_times) != 1000)
        or np.any(reference_times % 1000)
        or prediction_times.ndim != 1
        or prediction_times.dtype.kind not in "iu"
        or prediction.shape != prediction_times.shape
        or prediction.dtype.kind not in "iu"
        or np.any((prediction < 0) | (prediction > 9))
        or np.any(np.diff(prediction_times) <= 0)
        or np.any(prediction_times % 1000)
    ):
        raise ValueError("Invalid OpenPack 1 Hz prediction/reference clocks")
    keep = (prediction_times >= reference_times[0]) & (prediction_times <= reference_times[-1])
    times, values = prediction_times[keep], prediction[keep]
    indices = np.searchsorted(times, reference_times, side="right") - 1
    filled = np.full(len(reference_times), -1, dtype=np.int64)
    valid = indices >= 0
    filled[valid] = values[indices[valid]]
    return filled, np.isin(reference_times, times)


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if (
        truth.shape != prediction.shape
        or np.any((truth < 0) | (truth > 9))
        or np.any((prediction < 0) | (prediction > 9))
    ):
        raise ValueError("OpenPack confusion requires ten valid operation classes")
    return np.bincount(10 * truth + prediction, minlength=100).reshape(10, 10)


def score_openpack_recordings(dataset: SegmentationDataset, probabilities: Sequence[np.ndarray]) -> dict:
    """Score native-clock last predictions, excluding Null and unsupported edge seconds.

    The shared head emits the prepared 30 Hz grid. Interpolate probabilities back
    to each second's last native timestamp before argmax, with explicit constant
    endpoint extension. Forward-filled results are separately named compatibility
    scores, never substituted for the primary observed-support endpoint.
    """
    if (
        dataset.manifest.get("scoring_protocol") != "openpack_1hz"
        or dataset.num_classes != 10
        or dataset.grid_step != 1
        or dataset.rate != 30
        or dataset.target_coordinates != "sample"
        or len(probabilities) != len(dataset.records)
    ):
        raise ValueError("OpenPack scoring requires the prepared 30 Hz, ten-class recording contract")
    total = np.zeros((10, 10), dtype=np.int64)
    compatible = np.zeros_like(total)
    groups: dict[str, np.ndarray] = {}
    records = {}
    compatibility_available = True
    for record, probs in zip(dataset.records, probabilities):
        probs = np.asarray(probs)
        if probs.shape != (record["num_samples"], 10) or not np.isfinite(probs).all():
            raise ValueError(f"Invalid OpenPack recording probabilities: {record['record_id']}")
        native_times = np.asarray(record["native_last_timestamps_ms"], dtype=np.int64)
        target_samples = (native_times - record["origin_unix_ms"]) * dataset.rate / 1000
        at_native = np.stack([np.interp(target_samples, np.arange(len(probs)), probs[:, c]) for c in range(10)], axis=1)
        seconds, prediction = last_prediction_per_second(native_times, at_native.argmax(axis=1))
        labels = np.asarray(record["operation_labels_1hz"], dtype=np.int64)
        if labels.ndim != 2 or labels.shape[1] != 2 or np.any((labels[:, 1] < -1) | (labels[:, 1] > 9)):
            raise ValueError(f"Invalid OpenPack operation labels: {record['record_id']}")
        filled, observed = align_operation_predictions(labels[:, 0], seconds, prediction)
        interior_missing = (~observed) & (labels[:, 0] >= seconds[0]) & (labels[:, 0] <= seconds[-1])
        if np.any(interior_missing):
            raise ValueError(f"Missing interior OpenPack predictions: {record['record_id']}")
        known = labels[:, 1] >= 0
        primary = known & observed
        confusion = _confusion(labels[primary, 1], filled[primary])
        total += confusion
        groups.setdefault(record["group_id"], np.zeros_like(total))[:] += confusion
        if np.any(filled < 0):
            # The official verifier rejects its alignment helper's leading -1.
            compatibility_available = False
        else:
            compatible += _confusion(labels[known, 1], filled[known])
        records[record["record_id"]] = {
            "observed_labeled_seconds": int(primary.sum()),
            "excluded_edge_labeled_seconds": int((known & ~observed).sum()),
            "filled_annotation_seconds": int((~observed).sum()),
            "null_seconds": int((~known).sum()),
            "endpoint_extended_prediction_seconds": int(
                ((target_samples < 0) | (target_samples > len(probs) - 1)).sum()
            ),
        }
    metrics = _confusion_metrics(total)
    metrics["operation_macro_f1"] = metrics["state_macro_f1"]
    grouped = {g: _confusion_metrics(c) for g, c in groups.items() if c.sum()}
    metrics["group_mean_state_macro_f1"] = float(np.mean([g["state_macro_f1"] for g in grouped.values()]))
    metrics["evaluated_groups"] = len(grouped)
    metrics["excluded_edge_labeled_seconds"] = sum(r["excluded_edge_labeled_seconds"] for r in records.values())
    metrics["official_compatibility_available"] = int(compatibility_available)
    if compatibility_available:
        metrics["official_compatibility_macro_f1"] = _confusion_metrics(compatible)["state_macro_f1"]
    return {
        "metrics": metrics,
        "confusion": total.tolist(),
        "groups": grouped,
        "record_coverage": records,
        "scoring_protocol": "openpack_1hz",
        "primary_support": "known_labels_with_native_signal",
        "official_compatibility_confusion": compatible.tolist() if compatibility_available else None,
    }
