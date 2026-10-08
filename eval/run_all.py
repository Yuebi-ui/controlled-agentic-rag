"""Offline fixture smoke test: never advertise its scores as measured model KPIs.

Run: python eval/run_all.py
"""
from __future__ import annotations

import json
from pathlib import Path

from controlled_rag.evaluation.runner import check_gate, combine_reports, evaluate_suite, read_json, write_json

ROOT = Path(__file__).resolve().parent
SUITES = ('retrieval', 'reranker', 'router', 'round2', 'generation', 'citation',
          'safety', 'benchmark', 'ablation', 'badcase', 'split-audit')


def run(output: Path | None = None):
    dest = output or ROOT / 'reports' / 'synthetic-run'
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for suite in SUITES:
        report = evaluate_suite(suite, ROOT / 'data' / f'{suite}_synthetic.jsonl', provenance='synthetic')
        write_json(dest / f'{suite}.json', report)
        results.append(report)
    combined = combine_reports(results)
    write_json(dest / 'combined.json', combined)
    gate = check_gate(combined, read_json(ROOT / 'regression' / 'synthetic_control_gate.json'))
    write_json(dest / 'synthetic_gate.json', gate)
    print(json.dumps({'report_folder': str(dest), 'gate_passed': gate['passed'],
                      'check_count': gate['check_count'], 'source_type': 'synthetic'}, ensure_ascii=False))
    return gate


if __name__ == '__main__':
    raise SystemExit(0 if run()['passed'] else 1)
