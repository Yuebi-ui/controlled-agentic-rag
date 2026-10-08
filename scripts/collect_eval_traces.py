"""Metadata-only pipeline traces (synthetic demo; real factories are user-supplied).

python scripts/collect_eval_traces.py --cases eval/data/pipeline_cases_synthetic.jsonl \
  --output outputs/eval_trace --mode synthetic

Real usage requires --factory MODULE:FUNCTION. The function must return
(pipeline: ControlledRAGPipeline, request_factory: callable(id)->RequestContext).
Nothing in this module silently substitutes demo data for real sources.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from controlled_rag.demo import build_demo_pipeline, demo_request
from controlled_rag.evaluation.tracing import LabeledPipelineCase, collect_case, save_records


def _factory(spec: str):
    module, sep, name = spec.partition(':')
    if not sep or not module or not name:
        raise ValueError('factory must be in MODULE:FUNCTION format')
    function = getattr(importlib.import_module(module), name)
    return function()


def run(argv=None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['synthetic', 'real'], required=True)
    parser.add_argument('--factory', help='real-mode module:function returning (pipeline, request_factory)')
    args = parser.parse_args(argv)
    if args.mode == 'real':
        if not args.factory:
            parser.error('real mode requires --factory')
        pipeline, request_factory = _factory(args.factory)
    else:
        if args.factory:
            parser.error('synthetic mode must not load a user-supplied production factory')
        pipeline, request_factory = build_demo_pipeline(), demo_request
    cases = []
    with Path(args.cases).open(encoding='utf-8') as input_file:
        for idx, line in enumerate(input_file, 1):
            if line.strip():
                try:
                    cases.append(LabeledPipelineCase.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f'invalid case in line {idx}') from exc
    if not cases or len(set(c.id for c in cases)) != len(cases):
        raise ValueError('test cases missing or duplicate case IDs')
    traces = []
    for case in cases:
        request = request_factory(case.id)
        _, data = collect_case(pipeline, request, case)
        traces.append(data)
    counts = save_records(args.output, traces)
    metadata = {'source_type': args.mode, 'counts': counts,
                'model_quality_evaluated': False,
                'note': 'Synthetic mode uses mock generator, never Qwen3-DPO weights. Semantic labels must be independently supplied.'}
    (Path(args.output) / 'trace_manifest.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))
    return metadata


if __name__ == '__main__':
    run()
