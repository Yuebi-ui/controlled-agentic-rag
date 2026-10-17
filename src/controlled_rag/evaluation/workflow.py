"""End-to-end Agent workflow decisions, separate from answer quality and faithfulness.

Gold route/trigger labels come from annotated cases, never from model predictions.
Unanswerable-to-answer violations use corpus labels; gate violations use only
observed control-flow states. They must not be confused with semantic correctness.
"""
from __future__ import annotations

from .metrics import confusion_stats, divide, f1
from .schemas import WorkflowRecord


def evaluate_workflow(rows: list[WorkflowRecord]) -> dict:
    if not rows:
        raise ValueError('empty workflow dataset')
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate workflow case IDs')

    routed = [r for r in rows if r.predicted_round1_state is not None]
    routes = (confusion_stats([r.gold_round1_state for r in routed],
                              [r.predicted_round1_state for r in routed]) if routed else None)
    negatives = [r for r in routed if r.gold_round1_state != 'sufficient']
    false_suff = sum(r.predicted_round1_state == 'sufficient' for r in negatives)

    tp = sum(r.expected_round2 and r.actual_round2 for r in rows)
    fp = sum(not r.expected_round2 and r.actual_round2 for r in rows)
    fn = sum(r.expected_round2 and not r.actual_round2 for r in rows)
    tn = len(rows) - tp - fp - fn
    trigger_precision = divide(tp, tp + fp)
    trigger_recall = divide(tp, tp + fn)

    answered = [r for r in rows if r.final_status == 'answered']
    answered_correctly = sum(r.gold_answerable for r in answered)
    answerable = sum(r.gold_answerable for r in rows)
    answer_precision = divide(answered_correctly, len(answered))
    answer_recall = divide(answered_correctly, answerable)
    # Failed requests are not counted as correct abstentions.
    exact_decisions = sum(
        (r.gold_answerable and r.final_status == 'answered') or
        (not r.gold_answerable and r.final_status == 'abstained') for r in rows
    )

    violations: dict[str, list[str]] = {
        'answered_without_sufficient_gate': [],
        'sealed_without_sufficient_gate': [],
        'round2_after_sufficient_round1': [],
        'mandatory_round1_evidence_lost': [],
        'answered_gold_unanswerable': [],
    }
    for r in rows:
        if r.final_status == 'answered' and r.final_gate_state != 'sufficient':
            violations['answered_without_sufficient_gate'].append(r.id)
        if r.sealed_final_present and r.final_gate_state != 'sufficient':
            violations['sealed_without_sufficient_gate'].append(r.id)
        if r.actual_round2 and r.predicted_round1_state == 'sufficient':
            violations['round2_after_sufficient_round1'].append(r.id)
        if r.actual_round2 and (r.merged_evidence_keys is None or not
                set(r.mandatory_round1_keys).issubset(r.merged_evidence_keys)):
            violations['mandatory_round1_evidence_lost'].append(r.id)
        if r.final_status == 'answered' and not r.gold_answerable:
            violations['answered_gold_unanswerable'].append(r.id)

    rescue_eligible = [r for r in rows if r.actual_round2 and r.gold_answerable
                       and r.predicted_round1_state != 'sufficient']
    rescued = [r for r in rescue_eligible if r.final_gate_state == 'sufficient'
               and r.final_status == 'answered']
    return {
        'sample_count': len(rows),
        'route1': {
            'labeled_count': len(routed),
            'missing_predictions': len(rows) - len(routed),
            'classification': routes,
            'gold_insufficient_count': len(negatives),
            'false_sufficient_count': false_suff,
            'false_sufficient_rate': divide(false_suff, len(negatives)),
        },
        'round2_trigger': {
            'expected_count': tp + fn,
            'actual_count': tp + fp,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': trigger_precision,
            'recall': trigger_recall,
            'f1': f1(trigger_precision, trigger_recall),
            'accuracy': (tp + tn) / len(rows),
        },
        'answerability_decision': {
            'gold_answerable_count': answerable,
            'answered_count': len(answered),
            'correct_decision_count': exact_decisions,
            'accuracy': exact_decisions / len(rows),
            'answer_precision': answer_precision,
            'answer_recall': answer_recall,
            'answer_f1': f1(answer_precision, answer_recall),
            'failed_requests': sum(r.final_status == 'failed' for r in rows),
        },
        'rescue': {
            'eligible_triggered_answerable_count': len(rescue_eligible),
            'rescued_count': len(rescued),
            'rescue_rate': divide(len(rescued), len(rescue_eligible)),
        },
        'control_invariants': {
            'violation_counts': {key: len(value) for key, value in violations.items()},
            'violating_case_ids': violations,
            'all_hold': all(not value for value in violations.values()),
        },
        'interpretation': (
            'Answerability uses corpus-level gold and does not measure factual correctness. '
            'Synthetic traces cannot establish real-world agent accuracy, safety or latency.'
        ),
    }
