"""Explicit reference examples for OpenPack's wall-clock operation score."""

from types import SimpleNamespace

import numpy as np
import pytest

from trd.evaluation.openpack import align_operation_predictions, last_prediction_per_second, score_openpack_recordings

pytestmark = pytest.mark.unit


def recording_fixture(*, leading: bool = False) -> tuple[SimpleNamespace, list[np.ndarray]]:
    """Two seconds of observed signal, a trailing known label and ten classes."""
    labels = [[1000, 0], [2000, 1], [3000, 1]]
    if leading:
        labels.insert(0, [0, -1])
    dataset = SimpleNamespace(
        manifest={"scoring_protocol": "openpack_1hz"},
        num_classes=10,
        rate=30,
        grid_step=1,
        target_coordinates="sample",
        records=[
            {
                "record_id": "record",
                "group_id": "person",
                "num_samples": 60,
                "origin_unix_ms": 1000,
                "native_last_timestamps_ms": [1990, 2980],
                "operation_labels_1hz": labels,
            }
        ],
    )
    probabilities = np.zeros((60, 10), dtype=np.float32)
    probabilities[:31, 0], probabilities[31:, 1] = 1, 1
    return dataset, [probabilities]


def test_last_prediction_is_neither_majority_nor_rounded_timestamp():
    seconds, classes = last_prediction_per_second(
        np.array([1000, 1030, 1060, 1990, 2000, 2990]), np.array([0, 0, 0, 4, 7, 9])
    )
    assert seconds.tolist() == [1000, 2000]
    assert classes.tolist() == [4, 9]


def test_alignment_crops_before_forward_fill_and_preserves_leading_unknown():
    filled, observed = align_operation_predictions(
        np.arange(1000, 6000, 1000), np.array([0, 2000, 4000, 6000]), np.array([9, 2, 4, 8])
    )
    assert filled.tolist() == [-1, 2, 2, 4, 4]
    assert observed.tolist() == [False, True, False, True, False]


def test_primary_counts_observed_seconds_and_compatibility_reports_filled_edge():
    dataset, probabilities = recording_fixture()
    result = score_openpack_recordings(dataset, probabilities)
    assert result["metrics"]["labeled_frames"] == 2
    assert result["metrics"]["state_accuracy"] == 1
    assert result["metrics"]["operation_macro_f1"] == pytest.approx(0.2)  # All ten classes, absent classes zero.
    assert result["metrics"]["excluded_edge_labeled_seconds"] == 1
    assert result["metrics"]["official_compatibility_available"] == 1
    assert np.array(result["official_compatibility_confusion"]).sum() == 3
    assert result["record_coverage"]["record"]["endpoint_extended_prediction_seconds"] == 1


def test_leading_unknown_makes_official_verifier_incompatible_even_when_null():
    dataset, probabilities = recording_fixture(leading=True)
    result = score_openpack_recordings(dataset, probabilities)
    assert result["metrics"]["labeled_frames"] == 2
    assert result["metrics"]["official_compatibility_available"] == 0
    assert "official_compatibility_macro_f1" not in result["metrics"]
    assert result["official_compatibility_confusion"] is None


def test_missing_interior_signal_is_not_silently_forward_filled():
    dataset, probabilities = recording_fixture()
    dataset.records[0]["native_last_timestamps_ms"] = [1990, 3980]
    with pytest.raises(ValueError, match="Missing interior"):
        score_openpack_recordings(dataset, probabilities)


def test_null_is_excluded_but_picking_is_scored():
    dataset, probabilities = recording_fixture()
    dataset.records[0]["operation_labels_1hz"] = [[1000, 0], [2000, -1]]
    result = score_openpack_recordings(dataset, probabilities)
    assert result["metrics"]["labeled_frames"] == 1
    assert result["confusion"][0][0] == 1


@pytest.mark.parametrize("times", [[1000, 1000], [2000, 1000]])
def test_ambiguous_prediction_timestamps_fail(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        last_prediction_per_second(np.array(times), np.array([0, 1]))
