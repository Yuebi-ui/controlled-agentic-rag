"""Final-gate and Round2 measurements: never equate triggering with rescue."""
from __future__ import annotations

from .metrics import avg, divide
from .schemas import Round2Record


def evaluate_round2(rows: list[Round2Record]) -> dict:
    if not rows:
        raise ValueError('empty Round2 dataset')
    evaluated = [r for r in rows if r.retrieval_rounds >= 1]
    triggered = [r for r in evaluated if r.retrieval_rounds == 2]
    eligible = [r for r in triggered if r.gold_answerable and r.round1_state != 'sufficient']
    rescued = [r for r in eligible if r.round2_state == 'sufficient' and r.final_status == 'answered']
    violations = {
        'unsafe_answer_without_sufficient_final_gate': 0,
        'unauthorized_round2_after_sufficient_round1': 0,
        'mandatory_round1_evidence_lost': 0,
        'answered_unanswerable_gold': 0,
    }
    for r in rows:
        final_gate = r.round2_state if r.retrieval_rounds == 2 else r.round1_state
        if r.final_status == 'answered' and final_gate != 'sufficient':
            violations['unsafe_answer_without_sufficient_final_gate'] += 1
        if r.round1_state == 'sufficient' and r.retrieval_rounds == 2:
            violations['unauthorized_round2_after_sufficient_round1'] += 1
        if r.mandatory_round1_keys and not set(r.mandatory_round1_keys).issubset(r.merged_evidence_keys if r.merged_evidence_keys is not None else r.final_evidence_keys):
            violations['mandatory_round1_evidence_lost'] += 1
        if r.final_status == 'answered' and not r.gold_answerable:
            violations['answered_unanswerable_gold'] += 1
    return {
        'sample_count': len(rows), 'routed_count': len(evaluated),
        'round2_trigger_count': len(triggered), 'round2_trigger_rate': divide(len(triggered), len(evaluated)),
        'gold_answerable_trigger_count': len(eligible), 'rescued_answer_count': len(rescued),
        'rescue_rate_answerable_triggered': divide(len(rescued), len(eligible)),
        'rescue_rate_all_triggered': divide(len(rescued), len(triggered)),
        'mean_coverage_gain_triggered': avg([r.final_covered/r.requirement_count -
                                             r.round1_covered/r.requirement_count for r in triggered]),
        'mean_final_coverage_triggered': avg([r.final_covered/r.requirement_count for r in triggered]),
        'violations': violations,
        'all_control_invariants_hold': all(v == 0 for v in violations.values()),
    }
