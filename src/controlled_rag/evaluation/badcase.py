"""General regression bad-case suite: fixed expected behavior and severity.

Case IDs are stable across releases and failures can be linked to private
redacted traces. Do not move test cases to train when a model overfits them.
"""
from __future__ import annotations

from .schemas import BadCaseRecord


def evaluate_badcases(rows: list[BadCaseRecord]) -> dict:
    if not rows:
        raise ValueError('empty badcase dataset')
    failing = []
    by_category = {}
    for row in rows:
        violations=[]
        if row.actual_status != row.expected_status:
            violations.append('wrong_final_status')
        if row.expected_reason_codes and row.actual_reason_code not in row.expected_reason_codes:
            violations.append('wrong_reason_code')
        if row.actual_retrieval_rounds > row.max_allowed_rounds:
            violations.append('too_many_retrieval_rounds')
        if set(row.forbidden_citation_ids) & set(row.observed_citation_ids):
            violations.append('forbidden_citation')
        stats=by_category.setdefault(row.category, {'total':0, 'failures':0})
        stats['total']+=1
        if violations:
            stats['failures']+=1
            failing.append({'id':row.id,'category':row.category,
                            'severity':row.severity,'violations':violations})
    return {'sample_count':len(rows), 'passed_count':len(rows)-len(failing),
            'pass_rate':(len(rows)-len(failing))/len(rows),
            'critical_failed_count':sum(f['severity']=='critical' for f in failing),
            'high_failed_count':sum(f['severity']=='high' for f in failing),
            'failed_cases':failing, 'by_category':by_category}
