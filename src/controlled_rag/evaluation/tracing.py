"""Opt-in metadata-only pipeline trace for real/Mock evaluation harnesses.

Generated answers, full query, PDF contents, API keys, model weights, and ACL
principal IDs are intentionally absent. Source URIs are still sensitive: write
traces only to private `outputs/` and sanitize before publishing anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import PipelineResult, RequestContext
from ..orchestrator import ControlledRAGPipeline
from .schemas import (CitationRecord, RetrievalRecord, RerankerRecord,
                      Round2Record)


@dataclass
class TraceRecorder:
    events: dict[str, dict] = field(default_factory=dict)

    def record(self, event: str, payload: dict) -> None:
        if event in self.events:
            raise ValueError(f'repeated event in one pipeline run: {event}')
        self.events[event] = payload


class LabeledPipelineCase(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold_answerable: bool
    gold_parent_ids: list[str] = Field(default_factory=list)
    gold_image_ids: list[str] = Field(default_factory=list)


def collect_case(pipeline: ControlledRAGPipeline, request: RequestContext,
                 case: LabeledPipelineCase, *, observer: TraceRecorder | None = None
                 ) -> tuple[PipelineResult, dict[str, dict]]:
    """Collect supported per-stage records without fabricating semantic labels."""
    trace = observer if observer is not None else TraceRecorder()
    result = pipeline.run(case.query, request, observer=trace)
    ev = trace.events
    records: dict[str, dict] = {}
    if 'retrieval1' in ev and (case.gold_parent_ids or case.gold_image_ids):
        r1 = ev['retrieval1']
        records['retrieval_round1'] = RetrievalRecord(
            id=case.id, gold_parent_ids=case.gold_parent_ids,
            gold_image_ids=case.gold_image_ids,
            ranked_parent_ids=r1['parent_ids'], ranked_image_ids=r1['image_ids']).model_dump()
        if 'retrieval2' in ev:
            # Union in order of appearance. This reflects total candidate pool
            # across both retrievals, NOT the final reranker top-K.
            r2 = ev['retrieval2']
            parents = list(dict.fromkeys(r1['parent_ids'] + r2['parent_ids']))
            images = list(dict.fromkeys(r1['image_ids'] + r2['image_ids']))
        else:
            parents, images = r1['parent_ids'], r1['image_ids']
        records['retrieval_two_round_pool'] = RetrievalRecord(
            id=case.id, gold_parent_ids=case.gold_parent_ids,
            gold_image_ids=case.gold_image_ids,
            ranked_parent_ids=parents, ranked_image_ids=images).model_dump()
    if 'rerank_all1' in ev and case.gold_parent_ids:
        rank = ev['rerank_all1']
        records['reranker_round1'] = RerankerRecord(
            id=case.id, positive_ids=case.gold_parent_ids,
            before=rank['before'], after=rank['after'],
            after_scores=rank['after_scores']).model_dump()
    if 'route1' in ev:
        first = ev['route1']
        last = ev.get('route2', first)
        mandatory = first['covered_best_keys'] if 'route2' in ev else []
        records['round2'] = Round2Record(
            id=case.id, round1_state=result.route1, round2_state=result.route2,
            retrieval_rounds=result.retrieval_rounds, final_status=result.final_status,
            gold_answerable=case.gold_answerable,
            requirement_count=first['requirement_count'],
            round1_covered=first['requirement_count'] - len(first['missing_requirements']),
            final_covered=first['requirement_count'] - len(last['missing_requirements']),
            mandatory_round1_keys=mandatory,
            merged_evidence_keys=ev['round2_merged']['evidence_keys'] if 'round2_merged' in ev else None,
            final_evidence_keys=ev.get('final_candidate_keys', {}).get('evidence_keys', [])).model_dump()
    if 'sealed_final' in ev:
        final = ev['sealed_final']
        records['citation'] = CitationRecord(
            id=case.id, allowed_evidence_ids=final['allowed_evidence_ids'],
            canonical_evidence=final['canonical_evidence'],
            citations=[{**citation.model_dump(), 'semantic_supported': None}
                       for citation in result.citations],
        ).model_dump()
    return result, records


def save_records(root: str | Path, records: list[dict[str, dict]]) -> dict[str, int]:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    named = {}
    for record in records:
        for name, payload in record.items():
            named.setdefault(name, []).append(payload)
    for name, group in named.items():
        with (directory / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for item in group:
                f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    return {name: len(items) for name, items in named.items()}
