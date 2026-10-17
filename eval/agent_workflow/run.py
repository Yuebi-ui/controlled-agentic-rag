"""Agent workflow benchmark runner (gold labels + observed pipeline decisions)."""
from __future__ import annotations

import argparse
import json

from controlled_rag.demo import build_demo_pipeline, demo_request
from controlled_rag.evaluation.workflow_runner import (
    load_cases, load_factory, run_workflow_cases, save_workflow_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evidence-driven agent workflow benchmark')
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('synthetic', 'real'), required=True)
    parser.add_argument('--factory', help='real mode: module:function -> (pipeline, request_factory)')
    args = parser.parse_args(argv)
    if args.mode == 'real':
        if not args.factory:
            parser.error('real mode requires --factory')
        pipeline, request_factory = load_factory(args.factory)()
    else:
        if args.factory:
            parser.error('synthetic mode must not use --factory')
        pipeline, request_factory = build_demo_pipeline(), demo_request
    rows, _ = run_workflow_cases(pipeline, request_factory, load_cases(args.cases))
    report = save_workflow_report(args.output, rows, args.mode, args.cases)
    print(json.dumps({'output': args.output, 'source_type': args.mode,
                      'sample_count': report['metrics']['sample_count']}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
