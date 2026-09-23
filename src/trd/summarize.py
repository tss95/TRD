"""Aggregate paired principal results; scores and gains are percentage points."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(rows: list[dict]) -> list[dict]:
    """Require three complete native/TRD seed pairs for each reported cell."""
    groups = {}
    for row in rows:
        key = (row['dataset'], row['host'], row['readout'], row['metric'])
        mode, seed, score = row['mode'], int(row['seed']), float(row['score']) * 100
        if mode not in {'native', 'trd'} or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('Invalid arm or score')
        if int(row['updates']) != 200000:
            raise ValueError('Principal tables require the fixed 200k endpoint')
        pair = groups.setdefault(key, {'native': {}, 'trd': {}})[mode]
        if seed in pair:
            raise ValueError(f'Duplicate seed in {key}')
        pair[seed] = score
    output = []
    for (dataset, host, readout, metric), pair in sorted(groups.items()):
        if any(set(scores) != {42, 7, 123} for scores in pair.values()):
            raise ValueError('Each cell needs matched native/TRD seeds 42, 7 and 123')
        native = [pair['native'][seed] for seed in (42, 7, 123)]
        joint = [pair['trd'][seed] for seed in (42, 7, 123)]
        gains = [b - a for a, b in zip(native, joint)]
        output.append(
            {
                'dataset': dataset,
                'host': host,
                'readout': readout,
                'metric': metric,
                'native_mean': statistics.mean(native),
                'native_sd': statistics.stdev(native),
                'trd_mean': statistics.mean(joint),
                'trd_sd': statistics.stdev(joint),
                'gain_mean': statistics.mean(gains),
                'gain_sd': statistics.stdev(gains),
                'positive_pairs': sum(gain > 0 for gain in gains),
            }
        )
    if not output:
        raise ValueError('No results to summarize')
    return output


def main() -> None:
    """Summarize the reference CSV or JSON results from independent reruns."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', type=Path)
    source.add_argument('--results', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.csv:
        with args.csv.open() as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
        for file in args.results:
            result = json.loads(file.read_text())
            if result['split'] != 'test':
                raise ValueError('Principal tables require held-out test results')
            metric = 'typed_event_macro_f1_at_0.05s' if result['dataset'] == 'circor' else 'operation_macro_f1'
            for name, readout in result['readouts'].items():
                rows.append(
                    {
                        **result,
                        'readout': name,
                        'metric': metric,
                        'score': readout['metrics'][metric],
                        'updates': result['step'],
                    }
                )
    summary = summarize(rows)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print('| Dataset | Host | Readout | Native | +TRD | Gain | Positive |')
    print('| --- | --- | --- | ---: | ---: | ---: | ---: |')
    for row in summary:
        print(
            f"| {row['dataset']} | {row['host']} | {row['readout']} | "
            f"{row['native_mean']:.2f} ± {row['native_sd']:.2f} | "
            f"{row['trd_mean']:.2f} ± {row['trd_sd']:.2f} | {row['gain_mean']:+.2f} | {row['positive_pairs']}/3 |"
        )


if __name__ == '__main__':
    main()
