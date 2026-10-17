"""Run the same annotated cases through a real or deterministic synthetic Pipeline.

No answer text or corpus content is included in the emitted case records. The
synthetic factory exists for wiring tests only; a real factory must be supplied
explicitly for a real run.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Callable, Sequence

from pydantic import Field, model_validator

from ..contracts import RequestContext
from ..orchestrator import ControlledRAGPipeline
from .metrics import at_k
from .runner import load_jsonl, write_json
from .schemas import WorkflowRecord
from .tracing import LabeledPipelineCase, TraceRecorder, collect_case
from .workflow import evaluate_workflow


class WorkflowCase(LabeledPipelineCase):
    """Human-annotated Round1 and trigger-policy labels for an evaluation case."""
    gold_round1_state: str = Field(pattern='^(sufficient|partial|failed)$')
    expected_round2: bool

    @model_validator(mode='after')
    def check_labels(self):
        if self.gold_round1_state == 'sufficient' and self.expected_round2:
            raise ValueError('sufficient Round1 gold must not expect Round2')
        return self


def load_cases(path: str | Path) -> list[WorkflowCase]:
    return load_jsonl(path, WorkflowCase, 'workflow_cases')


def load_factory(spec: str) -> Callable:
    module, sep, name = spec.partition(':')
    if not sep or not module or not name or '.' in name:
        raise ValueError('factory must be module:function')
    target = getattr(importlib.import_module(module), name)
    if not callable(target):
        raise ValueError('factory must be callable')
    return target


def run_workflow_cases(
    pipeline: ControlledRAGPipeline,
    request_factory: Callable[[str], RequestContext],
    cases: Sequence[WorkflowCase],
) -> tuple[list[WorkflowRecord], dict[str, dict[str, float]]]:
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError('nonempty unique case IDs required')
    records: list[WorkflowRecord] = []
    per_case_metrics: dict[str, dict[str, float]] = {}
    for case in cases:
        trace = TraceRecorder()
        start = time.perf_counter()
        result, _ = collect_case(pipeline, request_factory(case.id), case, observer=trace)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ev = trace.events
        route1 = ev.get('route1', {})
        route2 = ev.get('route2', {})
        has_r2 = result.retrieval_rounds == 2
        record = WorkflowRecord(
            id=case.id,
            gold_answerable=case.gold_answerable,
            gold_round1_state=case.gold_round1_state,
            predicted_round1_state=result.route1,
            expected_round2=case.expected_round2,
            actual_round2=has_r2,
            retrieval_rounds=result.retrieval_rounds,
            final_gate_state=(route2 if has_r2 else route1).get('state'),
            final_status=result.final_status,
            reason_code=result.reason_code,
            sealed_final_present='sealed_final' in ev,
            mandatory_round1_keys=ev.get('rerank2', {}).get('mandatory_round1_keys', []),
            merged_evidence_keys=ev.get('round2_merged', {}).get('evidence_keys'),
            elapsed_ms=elapsed_ms,
        )
        records.append(record)
        metrics = {
            'answerability_decision_correct': float(
                (case.gold_answerable and result.final_status == 'answered') or
                (not case.gold_answerable and result.final_status == 'abstained')),
            'round2_trigger_correct': float(has_r2 == case.expected_round2),
            'answered_without_sufficient_gate': float(
                result.final_status == 'answered' and record.final_gate_state != 'sufficient'),
            'round2_triggered': float(has_r2),
        }
        if result.route1 is not None:
            metrics['round1_route_correct'] = float(result.route1 == case.gold_round1_state)
        r1 = ev.get('retrieval1')
        if r1 is not None:
            r2 = ev.get('retrieval2')
            if case.gold_parent_ids:
                ranked = list(dict.fromkeys(r1['parent_ids'] + (r2['parent_ids'] if r2 else [])))
                metrics['parent_candidate_recall_at_10'] = at_k(
                    ranked, set(case.gold_parent_ids), 10)['recall']
            if case.gold_image_ids:
                ranked = list(dict.fromkeys(r1['image_ids'] + (r2['image_ids'] if r2 else [])))
                metrics['image_candidate_recall_at_10'] = at_k(
                    ranked, set(case.gold_image_ids), 10)['recall']
        per_case_metrics[case.id] = metrics
    return records, per_case_metrics


def save_workflow_report(output: str | Path, records: list[WorkflowRecord],
                         source_type: str, input_path: str | Path,
                         *, context: dict | None = None) -> dict:
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'workflow_cases.jsonl').open('w', encoding='utf-8') as f:
        for record in records:
            f.write(record.model_dump_json() + '\n')
    report = {
        'suite': 'workflow',
        'provenance': {
            'source_type': source_type,
            'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            'model_quality_evaluated': False,
            'gold_labels': 'caller-supplied; not independently audited by runner',
            'context': context or {},
        },
        'metrics': evaluate_workflow(records),
    }
    write_json(dest / 'workflow_report.json', report)
    return report
