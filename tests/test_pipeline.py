"""A complete small train/fit/replay path for all three exported hosts."""

import json

import pytest
import torch

from conftest import tiny_config, prepared_data
from trd.checkpoint import load_checkpoint
from trd.data.windows import SegmentationDataset
from trd.evaluate import evaluate
from trd.probe import fit_heads
from trd.train import train

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('host', ['ts2vec', 'patchtst', 'pmt'])
@pytest.mark.parametrize('mode', ['native', 'trd'])
def test_train_fit_and_replay(tmp_path, monkeypatch, host, mode):
    config = tiny_config(host)
    if mode == 'native':
        config['trd']['weight'] = 0.0
    data = prepared_data(tmp_path / 'data')
    run, heads = tmp_path / 'run', tmp_path / 'heads'
    train(config, data, run, device='cpu')
    model, restored, step = load_checkpoint(run / 'step_2.pt', 'cpu')
    assert step == 2 and restored == config
    assert torch.isfinite(model.encode(torch.randn(2, 64, 1)).tokens).all()
    original = SegmentationDataset.__init__

    def no_test_reads(self, *args, **kwargs):
        assert kwargs['split'] != 'test', 'Test data entered fitting'
        original(self, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(SegmentationDataset, '__init__', no_test_reads)
        fit_heads(run / 'step_2.pt', data, heads, device='cpu', cache=tmp_path)
    result = evaluate(run / 'step_2.pt', data, heads, tmp_path / 'val.json', device='cpu', split='val', cache=tmp_path)
    validation = json.loads((heads / 'validation.json').read_text())
    for name, readout in result['readouts'].items():
        assert readout['metrics'] == validation[name]['metrics']
    test = evaluate(run / 'step_2.pt', data, heads, tmp_path / 'test.json', device='cpu', cache=tmp_path)
    assert len(test['readouts']) == 4 and test['split'] == 'test'
    with pytest.raises(ValueError, match='another checkpoint'):
        evaluate(run / 'step_0.pt', data, heads, tmp_path / 'wrong.json', device='cpu', cache=tmp_path)
