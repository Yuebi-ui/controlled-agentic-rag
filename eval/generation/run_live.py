"""Run Base/SFT/DPO endpoints on identical SEALED input, produce JSONL scores.

Requires real model endpoints; it never starts a fake vLLM service.
Inputs are private and MUST be sourced from approved final_evidence snapshots.
Correctness/faithfulness stay NULL until independently annotated.

python eval/generation/run_live.py --variant dpo \
  --input path/to/private_sealed_prompts.jsonl --output outputs/predictions_dpo.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_answer import (EvidenceItem, GenerationRequest,
                               GenerationContractError, ModelUnavailableError,
                               create_answer_model)
from controlled_rag.evaluation.schemas import GenerationRecord


def collect(adapter, data: list[dict], variant: str) -> list[dict]:
    """An unavailable endpoint aborts: never record it as invalid JSON metric."""
    output = []
    seen = set()
    for item in data:
        uid = item['id']
        if uid in seen:
            raise ValueError(f'duplicate generation query ID {uid}')
        seen.add(uid)
        request = GenerationRequest(
            original_query=item['original_query'],
            atomic_requirements=tuple(item['atomic_requirements']),
            final_evidence=tuple(EvidenceItem(**e) for e in item['final_evidence']))
        try:
            answer = adapter.generate(request)
        except GenerationContractError:
            prediction = {'schema_valid': False, 'predicted_abstain': None}
        # ModelUnavailableError is intentionally NOT swallowed, including missing DPO weights.
        else:
            prediction = {'schema_valid': True, 'predicted_abstain': answer.abstain}
        output.append(GenerationRecord(id=uid, model_variant=variant,
                    gold_answerable=item['gold_answerable'], correctness=None,
                    faithfulness=None, **prediction).model_dump())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['base','sft','dpo'], default='dpo')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    data=[]
    for idx, line in enumerate(Path(args.input).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                data.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f'bad JSON line {idx}') from exc
    if not data:
        raise ValueError('empty generation input')
    try:
        predictions = collect(create_answer_model(variant=args.variant), data, args.variant)
    except ModelUnavailableError as exc:
        raise SystemExit(f'Generation aborted; no metrics were written: {exc}') from exc
    outfile=Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in predictions),
                       encoding='utf-8')
    print(json.dumps({'variant': args.variant, 'predictions': len(predictions),
                      'semantic_annotations': 0, 'output': str(outfile)},ensure_ascii=False))


if __name__ == '__main__':
    main()
