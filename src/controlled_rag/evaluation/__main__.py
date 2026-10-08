"""python -m controlled_rag.evaluation --suite retrieval --input file.jsonl ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import (SUITES, check_gate, evaluate_suite, read_json, write_json)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Controlled RAG evaluation (explicit data provenance)')
    parser.add_argument('--suite', required=True, choices=[*SUITES, 'release'])
    parser.add_argument('--input', required=True, help='JSONL dataset, or combined report for release')
    parser.add_argument('--output', required=True, help='JSON output path')
    parser.add_argument('--source', default='synthetic',
                        choices=['synthetic', 'real_unverified', 'real_audited'])
    parser.add_argument('--model-version')
    parser.add_argument('--index-version')
    parser.add_argument('--ablation-baseline', default='baseline')
    parser.add_argument('--rules', help='Release gate rules JSON')
    parser.add_argument('--baseline', help='Baseline combined JSON report for delta checks')
    args = parser.parse_args(argv)
    if args.suite == 'release':
        if not args.rules:
            parser.error('--suite release requires --rules')
        report = check_gate(read_json(args.input), read_json(args.rules),
                            read_json(args.baseline) if args.baseline else None)
    else:
        report = evaluate_suite(args.suite, args.input, provenance=args.source,
                                model_version=args.model_version, index_version=args.index_version,
                                ablation_baseline=args.ablation_baseline)
    write_json(args.output, report)
    print(json.dumps({'suite': args.suite, 'output': args.output,
                      'passed': report['passed'] if args.suite == 'release' else None}, ensure_ascii=False))
    return 1 if args.suite == 'release' and not report['passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
