"""End-to-end numerical consistency, boundary behaviors and tamper regression."""
from __future__ import annotations


import pytest

from controlled_rag.evaluation.e2e import (
    summarize_end_to_end, summarize_outcome_ablation, validate_rows,
)



def row(uid, variant, *, gold=True, status="answered", gate="sufficient"):
    return dict(id=uid, variant=variant, gold_answerable=gold,
                final_status=status, final_gate_state=gate,
                sealed_final_present=variant == "controlled_full" and status == "answered",
                generation_called=status == "answered",
                answer_correct=True if status == "answered" else None,
                claim_supported=True if status == "answered" else None,
                citation_valid=True if status == "answered" else None,
                latency_ms=100, retrieval_rounds=1)




def test_pairwise_success_and_correct_refusal():
    cases = [row("q1", "basic_rag", gold=True, status="abstained"),
             row("q2", "basic_rag", gold=False, status="answered"),
             row("q1", "controlled_full", gold=True),
             row("q2", "controlled_full", gold=False, status="abstained")]
    result = summarize_end_to_end(cases, resamples=20)
    assert result["variants"]["basic_rag"]["grounded_success_per_input"] == 0
    assert result["variants"]["controlled_full"]["grounded_success_per_input"] == 1
    assert result["paired"]["grounded_success_per_input"]["delta_full_minus_basic"] == 1


def test_strict_empty_denominators():
    records = [row("q1", "basic_rag", gold=False, status="abstained"),
               row("q1", "controlled_full", gold=False, status="abstained")]
    result = summarize_end_to_end(records, resamples=15)
    assert result["variants"]["controlled_full"]["correct_answer_per_answerable"] is None
    assert result["variants"]["controlled_full"]["answer_faithfulness_given_answered"] is None
    assert result["variants"]["controlled_full"]["correct_refusal_per_unanswerable"] == 1


@pytest.mark.parametrize("change,expected", [
    (lambda xs: xs.pop(), "unpaired"),
    (lambda xs: xs.append(dict(xs[0])), "duplicate"),
    (lambda xs: xs[1].update(final_gate_state="failed"), "gate"),
    (lambda xs: xs[1].update(citation_valid=None), "citation"),
    (lambda xs: xs[1].update(latency_ms=float("nan")), "latency"),
    (lambda xs: xs[1].update(gold_answerable=False), "inconsistent"),
    (lambda xs: xs[1].update(retrieval_rounds=3), "retrieval_rounds"),
])
def test_bad_paired_inputs_rejected(change, expected):
    rows = [row("q", "basic_rag"), row("q", "controlled_full")]
    change(rows)
    with pytest.raises(ValueError, match=expected):
        summarize_end_to_end(rows, resamples=5)


def test_unsupported_claim_not_counted_as_grounded_success():
    rows = [row("q", "basic_rag"), row("q", "controlled_full")]
    rows[1]["claim_supported"] = False
    summary = summarize_end_to_end(rows, resamples=10)
    assert summary["variants"]["controlled_full"]["overall_decision_correct"] == 1
    assert summary["variants"]["controlled_full"]["grounded_success_per_input"] == 0


def test_ablation_rejects_duplicate_and_unpaired_id():
    base = [dict(id="q1", variant="controlled_full", grounded_success=True),
            dict(id="q1", variant="without_round2", grounded_success=False)]
    report = summarize_outcome_ablation(base, resamples=5)
    assert report["variants"]["without_round2"]["delta_full_minus_variant"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        summarize_outcome_ablation(base + [dict(base[0])], resamples=5)
    with pytest.raises(ValueError, match="unpaired"):
        summarize_outcome_ablation(base + [
            dict(id="q2", variant="controlled_full", grounded_success=True)], resamples=5)






