"""Model comparisons. Correctness/faithfulness require external labels.

Never score a generated answer as supported merely because a JSON citation exists.
Missing annotations are excluded and annotation coverage is reported.
"""
from __future__ import annotations

from .metrics import divide, f1
from .schemas import GenerationRecord


def _per_variant(rows: list[GenerationRecord]) -> dict:
    valid = [r for r in rows if r.schema_valid]
    answered = [r for r in valid if r.predicted_abstain is False]
    abstain_tp = sum(r.predicted_abstain is True and not r.gold_answerable for r in valid)
    abstain_fp = sum(r.predicted_abstain is True and r.gold_answerable for r in valid)
    abstain_fn = sum(r.predicted_abstain is False and not r.gold_answerable for r in valid)
    ap = divide(abstain_tp, abstain_tp + abstain_fp)
    ar = divide(abstain_tp, abstain_tp + abstain_fn)
    correct = [r.correctness for r in answered if r.correctness is not None]
    faithful = [r.faithfulness for r in answered if r.faithfulness is not None]
    return {
        'sample_count': len(rows), 'schema_valid_count': len(valid),
        'json_validity_rate': len(valid)/len(rows),
        'answered_count': len(answered), 'model_abstained_count': len(valid) - len(answered),
        'abstention_precision': ap, 'abstention_recall': ar, 'abstention_f1': f1(ap, ar),
        'unsafe_answers_on_unanswerable': abstain_fn,
        'false_abstentions_on_answerable': abstain_fp,
        'correctness_annotated_count': len(correct),
        'correctness_rate_annotated': divide(sum(correct), len(correct)),
        'faithfulness_annotated_count': len(faithful),
        'faithfulness_rate_annotated': divide(sum(faithful), len(faithful)),
    }


def evaluate_generation(rows: list[GenerationRecord]) -> dict:
    if not rows:
        raise ValueError('empty generation dataset')
    seen = set()
    grouped: dict[str, list[GenerationRecord]] = {}
    for row in rows:
        key = (row.model_variant, row.id)
        if key in seen:
            raise ValueError(f'duplicate model/sample pair: {key}')
        seen.add(key)
        grouped.setdefault(row.model_variant, []).append(row)
    ids = {variant: {r.id for r in value} for variant, value in grouped.items()}
    shared = set.intersection(*ids.values()) if ids else set()
    # Comparisons require same samples. Showing shared IDs does NOT imply
    # equal annotation coverage; use per-model counts when interpreting rates.
    return {
        'sample_count': len(rows),
        'shared_sample_count_across_reported_models': len(shared),
        'model_sample_ids_identical': all(x == shared for x in ids.values()),
        'models': {variant: _per_variant(data) for variant, data in sorted(grouped.items())},
    }
