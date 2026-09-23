"""Prepare public CirCor or OpenPack source files for the paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trd.data.circor import extract_circor_archive, prepare_circor
from trd.data.openpack import prepare_openpack


def main() -> None:
    """Convert downloaded source files into normalized recordings and a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['circor', 'openpack'])
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--archive', type=Path, help='Original CirCor 1.0.3 ZIP')
    parser.add_argument('--extract-to', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare = prepare_circor if args.dataset == 'circor' else prepare_openpack
    source = args.source
    if args.archive is not None:
        if args.dataset != 'circor' or args.extract_to is None:
            parser.error('--archive requires CirCor and a new --extract-to directory')
        source = extract_circor_archive(args.archive, args.extract_to)
    prepare(source, args.output)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    print(json.dumps({'dataset': args.dataset, 'recordings': len(manifest['records'])}))


if __name__ == '__main__':
    main()
