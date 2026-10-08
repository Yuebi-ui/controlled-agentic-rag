"""3-way Evidence Router performance + safety-specific false sufficient."""
from __future__ import annotations

from .metrics import ROUTES, avg, confusion_stats, divide, expected_calibration_error, multiclass_brier
from .schemas import RouterRecord


def evaluate_router(rows: list[RouterRecord]) -> dict:
    if not rows:
        raise ValueError('empty router dataset')
    gold = [r.gold_state for r in rows]
    pred = [r.predicted_state for r in rows]
    result = {'sample_count': len(rows), **confusion_stats(gold, pred)}
    wrong_sufficient = sum(g != 'sufficient' and p == 'sufficient' for g, p in zip(gold, pred))
    negative = sum(g != 'sufficient' for g in gold)
    result['false_sufficient_count'] = wrong_sufficient
    result['insufficient_gold_count'] = negative
    result['false_sufficient_rate'] = divide(wrong_sufficient, negative)
    result['false_sufficient_all_rate'] = wrong_sufficient / len(rows)
    raw = [r for r in rows if r.raw_model_state is not None]
    result['raw_model_sample_count'] = len(raw)
    result['raw_false_sufficient_rate'] = divide(
        sum(r.gold_state != 'sufficient' and r.raw_model_state == 'sufficient' for r in raw),
        sum(r.gold_state != 'sufficient' for r in raw))
    confident = [r for r in rows if r.probabilities is not None]
    probs = [tuple(r.probabilities[c] for c in ROUTES) for r in confident]
    truth = [r.gold_state for r in confident]
    result['calibration_sample_count'] = len(confident)
    result['brier_multiclass_unscaled'] = multiclass_brier(probs, truth)
    result['ece_toplabel_10bin'] = expected_calibration_error(probs, truth)
    return result
