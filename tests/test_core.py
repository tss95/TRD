"""Scientific invariants for the waveform manipulation and auxiliary loss."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from trd.corruption import corrupt_with_reversed_segment
from trd.loss import TemporalReversalLoss
from trd.random import preserve_rng, seed_all
from trd.targets import sample_reversal, token_labels_from_spans
from trd.hosts import build_host
from conftest import tiny_config

pytestmark = pytest.mark.unit


def test_joint_channel_reversal_and_linear_fade():
    x = torch.arange(60, dtype=torch.float32).reshape(1, 20, 3)
    changed, starts = corrupt_with_reversed_segment(x, 10, 3, starts=torch.tensor([4]))
    expected = x.clone()
    alpha = torch.tensor([0, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 0])[:, None]
    expected[0, 4:14] = alpha * x[0, 4:14].flip(0) + (1 - alpha) * x[0, 4:14]
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)
    assert starts.tolist() == [4]


def test_loss_excludes_padding_and_reaches_encoder():
    features = torch.randn(2, 8, 4, requires_grad=True)
    labels = torch.zeros(2, 8)
    labels[:, 2:4] = 1
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, -3:] = False
    head = TemporalReversalLoss(4)
    loss = head(features, labels, valid)
    expected = F.binary_cross_entropy_with_logits(head.head(features[valid]).squeeze(-1), labels[valid])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert features.grad[valid].abs().sum() > 0
    assert features.grad[~valid].abs().sum() == 0


def test_invalid_and_noncontiguous_support_is_rejected():
    x = torch.randn(2, 20, 1)
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[0, 10] = False
    with pytest.raises(ValueError, match='contiguous'):
        sample_reversal(x, 6, 1, valid)
    with pytest.raises(ValueError):
        sample_reversal(x, 6, 3)


def test_pmt_legacy_grid_includes_odd_patch_convention():
    labels = token_labels_from_spans(torch.tensor([0, 5]), 6, 5, 3, 5)
    # Nominal helper intervals: [0,4), [3,7), [6,10), [9,13), [12,16).
    assert labels.tolist() == [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]]


def test_ts2vec_auxiliary_restores_random_streams_and_native_loss():
    native = tiny_config('ts2vec')
    native['trd']['weight'] = 0.0
    seed_all(11)
    a = build_host(native).train()
    joint = tiny_config('ts2vec')
    seed_all(11)
    b = build_host(joint).train()
    x = torch.randn(2, 64, 1)
    seed_all(99)
    first = a(x)
    torch_after, numpy_after = torch.get_rng_state(), np.random.get_state()
    seed_all(99)
    second = b(x)
    torch.testing.assert_close(first['native_loss'], second['native_loss'], rtol=0, atol=0)
    assert torch.equal(torch_after, torch.get_rng_state())
    assert np.array_equal(numpy_after[1], np.random.get_state()[1])
    with pytest.raises(RuntimeError), preserve_rng():
        torch.rand(10)
        raise RuntimeError('exercise restoration')
    assert torch.equal(torch_after, torch.get_rng_state())
