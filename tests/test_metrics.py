"""Event matching, overlapping windows and physical feature coordinates."""

import pytest
import torch

from trd.data.temporal import match_event_onsets, recording_windows
from trd.evaluation.probe import align_dense_features
from trd.features import BackboneOutput

pytestmark = pytest.mark.unit


def test_event_matching_is_one_to_one_and_maximal():
    assert match_event_onsets([1.0], [0.98, 1.02], tolerance_seconds=0.05)[:3] == (1, 1, 0)
    assert match_event_onsets([1.0, 1.06], [0.95, 1.02], tolerance_seconds=0.05)[:3] == (2, 0, 0)
    assert match_event_onsets([], [], tolerance_seconds=0.05)[:3] == (0, 0, 0)
    with pytest.raises(ValueError, match='nonnegative'):
        match_event_onsets([-1.0], [], tolerance_seconds=0.05)


def test_ssl_tail_preserves_full_observed_window():
    assert recording_windows(153, length=100, stride=100, grid_step=10) == [(0, 100), (60, 93)]
    assert recording_windows(153, length=100, stride=100, allow_padding=False) == [(0, 100), (53, 100)]
    with pytest.raises(ValueError, match='full observed'):
        recording_windows(50, length=100, stride=100, allow_padding=False)


def test_feature_alignment_keeps_physical_coordinates():
    positions = torch.tensor([0.0, 4.0, 8.0])
    output = BackboneOutput(positions[None, :, None], None, None, None, None, sample_positions=positions)
    aligned = align_dense_features(output, input_length=10, grid_step=2)
    torch.testing.assert_close(aligned.flatten(), torch.tensor([0.5, 2.5, 4.5, 6.5, 8.0]))
    samples = align_dense_features(output, input_length=10, grid_step=2, sample_coordinates=True)
    torch.testing.assert_close(samples.flatten(), torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0]))
    bad = BackboneOutput(output.tokens, None, None, torch.tensor([[True, False, True]]), None)
    with pytest.raises(ValueError, match='invalid'):
        align_dense_features(bad, input_length=10, grid_step=2)
