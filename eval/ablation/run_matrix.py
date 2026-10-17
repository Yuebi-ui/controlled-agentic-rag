"""Executable, paired Pipeline ablation CLI."""
from __future__ import annotations

import argparse
import json

from controlled_rag.evaluation.ablation_runner import run_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run paired RAG pipeline ablations')
    parser.add_argument('--cases', required=True)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', required=True, choices=('synthetic', 'real'))
    parser.add_argument('--factory', help='real mode module:function -> (pipeline, request_factory)')
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    if args.mode == 'real' and not args.factory:
        parser.error('real mode requires --factory')
    if args.mode == 'synthetic' and args.factory:
        parser.error('synthetic mode does not accept --factory')
    report = run_matrix(cases_path=args.cases, matrix_path=args.matrix,
                        output=args.output, mode=args.mode, factory=args.factory,
                        bootstraps=args.bootstraps, seed=args.seed)
    print(json.dumps({'output': args.output, 'variants': list(report['variants']),
                      'source_type': args.mode}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
