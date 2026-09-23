"""Reference-table accounting and paired uncertainty calculations."""

import csv
from pathlib import Path

import pytest
from trd.summarize import summarize

pytestmark = pytest.mark.unit


def test_reference_results_recover_principal_paired_gains():
    source = Path(__file__).resolve().parents[1] / 'results/reference_scores.csv'
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 144
    cells = summarize(rows)
    primary = {(r['dataset'], r['host']): r for r in cells if r['readout'] in {'linear_lp0.05', 'linear_min5'}}
    assert len(primary) == 6
    expected = {
        ('circor', 'ts2vec'): 33.86,
        ('circor', 'patchtst'): 14.23,
        ('circor', 'pmt'): 47.32,
        ('openpack', 'ts2vec'): 2.25,
        ('openpack', 'patchtst'): 5.32,
        ('openpack', 'pmt'): 18.15,
    }
    for key, gain in expected.items():
        assert primary[key]['gain_mean'] == pytest.approx(gain, abs=0.005)
        assert primary[key]['positive_pairs'] == 3
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + [rows[0]])
    with pytest.raises(ValueError, match='matched'):
        summarize(rows[1:])
