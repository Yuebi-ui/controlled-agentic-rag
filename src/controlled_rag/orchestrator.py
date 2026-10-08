"""Explicit two-round, fail-closed Controlled Agentic RAG state machine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .citations import CitationGuard, Renderer
from .contracts import (Evidence, EvidenceSlot, FinalEvidence, IndexManifest,
                        IntegrityError, ModelOutputError, PipelineResult,
                        RequestContext, RouteState)
from .generation import AnswerModel
from .retrieval import RetrievalEngine, assert_valid_evidence, group_dedup
from .routing import Decomposer, EvidenceRouter, Rewriter, RouterDecision, Scorer


@dataclass(frozen=True)
class PipelineConfig:
    rerank_top_k: int = 5
    round1_retain: int = 5
    round2_pool_cap: int = 20
    max_retrieval_rounds: int = 2
    max_atomic_requirements: int = 8  # configurable serving guardrail, NOT historical param

    def __post_init__(self):
        if self.max_retrieval_rounds != 2:
            raise ValueError("MAX_RETRIEVAL_ROUNDS is structurally fixed at 2")
        if min(self.rerank_top_k, self.round1_retain, self.round2_pool_cap) < 1:
            raise ValueError("positive RAG candidate caps required")
        if self.round2_pool_cap < self.rerank_top_k:
            raise ValueError("round2_pool_cap cannot be less than rerank_top_k")
        if self.max_atomic_requirements < 1:
            raise ValueError("max_atomic_requirements must be positive")


def merge_round2(round1_useful: Sequence[Evidence], round2: Sequence[Evidence],
                 cap: int, mandatory_keys: Sequence[str] = ()) -> list[Evidence]:
    """Preserve covered-R1 evidence, dedup by Evidence Group and bound pool.

    Mandatory evidence is placed before generic top-K retained evidence.
    Integrity/version checks happen in the caller after merge.
    """
    lookup = {e.evidence_key: e for e in round1_useful}
    required = []
    for key in mandatory_keys:
        if key not in lookup:
            raise IntegrityError("covered Round1 evidence disappeared")
        required.append(lookup[key])
    if len(group_dedup(required, len(required) or 1)) > cap:
        raise IntegrityError("mandatory evidence exceeds round2 budget")
    return group_dedup([*required, *round1_useful, *round2], cap)


class ControlledRAGPipeline:
    def __init__(self, retrieval: RetrievalEngine, scorer: Scorer,
                 router: EvidenceRouter, decomposer: Decomposer,
                 rewriter: Rewriter, generator: AnswerModel,
                 config: PipelineConfig | None = None):
        self.retrieval = retrieval
        self.scorer = scorer
        self.router = router
        self.decomposer = decomposer
        self.rewriter = rewriter
        self.generator = generator
        self.config = config or PipelineConfig()
        self.guard = CitationGuard()
        self.renderer = Renderer()

    def _rank(self, query: str, evidences: Sequence[Evidence], *, observer=None,
              stage: str = 'rerank') -> tuple[list[Evidence], list[float]]:
        if not evidences:
            return [], []
        scores = self.scorer.score_many(query, evidences)
        if len(scores) != len(evidences) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in scores):
            raise IntegrityError("reranker returned invalid scores")
        ordered = sorted(zip(evidences, scores), key=lambda item: item[1], reverse=True)
        if observer is not None:
            observer.record(stage, {
                'before': [e.evidence_key for e in evidences],
                'after': [e.evidence_key for e, _ in ordered],
                'after_scores': {e.evidence_key: float(score) for e, score in ordered},
            })
        ordered = ordered[:self.config.rerank_top_k]
        return [e for e, _ in ordered], [float(s) for _, s in ordered]

    def _make_final(self, request: RequestContext, ranked: Sequence[Evidence]) -> FinalEvidence:
        for evidence in ranked:
            assert_valid_evidence(evidence, request, self.retrieval.manifest)
        if not ranked:
            raise IntegrityError("cannot seal empty final evidence")
        if len({e.evidence_group_id for e in ranked}) != len(ranked):
            raise IntegrityError("duplicate evidence group in final evidence")
        return FinalEvidence(
            request=request,
            manifest=self.retrieval.manifest,
            slots=tuple(EvidenceSlot(f"E{i}", e) for i, e in enumerate(ranked, 1)),
        )

    @staticmethod
    def _result(request: RequestContext, status: str, rounds: int, reason: str,
                answer: str = "", citations=None,
                route1: RouteState | None = None, route2: RouteState | None = None,
                degraded: set[str] | None = None) -> PipelineResult:
        return PipelineResult(
            request_id=request.request_id, index_version=request.index_version,
            final_status=status, answer=answer, citations=citations or [],
            retrieval_rounds=rounds, route1=route1, route2=route2,
            reason_code=reason, degraded_branches=sorted(degraded or set()),
        )

    def run(self, query: str, request: RequestContext, *, observer=None) -> PipelineResult:
        """Optional structured offline observer; never stores raw evidence text."""
        result = self._run(query, request, observer=observer)
        if observer is not None:
            observer.record('result', {'request_id': request.request_id,
                       'final_status': result.final_status,
                       'reason_code': result.reason_code,
                       'retrieval_rounds': result.retrieval_rounds,
                       'route1': result.route1, 'route2': result.route2})
        return result

    def _run(self, query: str, request: RequestContext, *, observer=None) -> PipelineResult:
        def record(stage, payload):
            if observer is not None:
                observer.record(stage, payload)

        if not query.strip():
            return self._result(request, "abstained", 0, "EMPTY_QUERY", "请提供具体的问题。")
        route1: RouteState | None = None
        route2: RouteState | None = None
        rounds = 0
        degraded: set[str] = set()
        try:
            # Pin index_version and verify principal before contacting any retriever.
            manifest: IndexManifest = self.retrieval.manifest
            if (manifest.index_version != request.index_version or
                    manifest.knowledge_base_id != request.knowledge_base_id or
                    request.knowledge_base_id not in request.allowed_kb_ids):
                raise IntegrityError("unauthorized or stale manifest")
            raw_requirements = self.decomposer.decompose(query)
            if isinstance(raw_requirements, str):
                raise IntegrityError("decomposer must return a sequence of requirement strings")
            if any(not isinstance(x, str) for x in raw_requirements):
                raise IntegrityError("non-string atomic requirement")
            requirements = tuple(dict.fromkeys(x.strip() for x in raw_requirements if x.strip()))
            if len(requirements) > self.config.max_atomic_requirements:
                raise IntegrityError("atomic requirement count exceeds configured cap")
            if not requirements:
                return self._result(request, "abstained", 0, "NO_ATOMIC_REQUIREMENTS", "问题信息不足。")

            batch1 = self.retrieval.retrieve(query, request)
            rounds = 1
            degraded.update(batch1.degraded_branches)
            record('retrieval1', {'parent_ids': [e.evidence_key for e in batch1.evidences if e.kind == 'text_parent'],
                                  'image_ids': [e.evidence_key for e in batch1.evidences if e.kind == 'image'],
                                  'degraded_branches': list(batch1.degraded_branches)})
            ranked1, scores1 = self._rank(query, batch1.evidences, observer=observer, stage='rerank_all1')
            record('rerank1', {'before': [e.evidence_key for e in batch1.evidences],
                               'after_top_k': [e.evidence_key for e in ranked1],
                               'top_k_scores': scores1})
            decision1 = self.router.route(requirements, scores1, ranked1)
            record('route1', {'state': decision1.state, 'raw_state': decision1.raw_model_state,
                              'feature_values': list(decision1.features),
                              'missing_requirements': list(decision1.missing_requirements),
                              'covered_best_keys': list(decision1.covered_best_keys),
                              'requirement_count': len(requirements)})
            route1 = decision1.state
            ranked = ranked1
            decision = decision1

            if decision.state != "sufficient":
                if decision.state == "partial" and decision.missing_requirements:
                    rewritten = self.rewriter.rewrite_missing(query, decision.missing_requirements)
                else:
                    rewritten = self.rewriter.rewrite_failed(query)
                if not rewritten.strip():
                    raise IntegrityError("round2 rewriter produced empty query")
                batch2 = self.retrieval.retrieve(rewritten, request)
                rounds = 2
                degraded.update(batch2.degraded_branches)
                record('retrieval2', {'parent_ids': [e.evidence_key for e in batch2.evidences if e.kind == 'text_parent'],
                                      'image_ids': [e.evidence_key for e in batch2.evidences if e.kind == 'image'],
                                      'degraded_branches': list(batch2.degraded_branches)})
                # Retain strong R1 and evidence that covered any original requirement.
                useful = ranked1[:self.config.round1_retain]
                useful_keys = {e.evidence_key for e in useful}
                required_keys = set(decision1.covered_best_keys)
                for evidence in ranked1:
                    if evidence.evidence_key in required_keys and evidence.evidence_key not in useful_keys:
                        useful.append(evidence)
                        useful_keys.add(evidence.evidence_key)
                merged = merge_round2(useful, batch2.evidences,
                                      self.config.round2_pool_cap, decision1.covered_best_keys)
                record('round2_merged', {'evidence_keys': [e.evidence_key for e in merged]})
                # Crucial: rerank on ORIGINAL query, rerun gate on ORIGINAL requirements.
                ranked, scores2 = self._rank(query, merged, observer=observer, stage='rerank_all2')
                record('rerank2', {'before': [e.evidence_key for e in merged],
                                   'after_top_k': [e.evidence_key for e in ranked],
                                   'top_k_scores': scores2,
                                   'mandatory_round1_keys': list(decision1.covered_best_keys)})
                decision = self.router.route(requirements, scores2, ranked)
                record('route2', {'state': decision.state, 'raw_state': decision.raw_model_state,
                                  'feature_values': list(decision.features),
                                  'missing_requirements': list(decision.missing_requirements),
                                  'covered_best_keys': list(decision.covered_best_keys),
                                  'requirement_count': len(requirements)})
                route2 = decision.state

            record('final_candidate_keys', {'evidence_keys': [e.evidence_key for e in ranked],
                                            'covered_requirement_count': len(requirements) - len(decision.missing_requirements)})
            if decision.state != "sufficient":
                return self._result(request, "abstained", rounds, "EVIDENCE_INSUFFICIENT",
                                    "现有手册证据不足以完整回答该问题。", route1=route1,
                                    route2=route2, degraded=degraded)

            sealed = self._make_final(request, ranked)
            record('sealed_final', {'allowed_evidence_ids': [slot.evidence_id for slot in sealed.slots],
                     'canonical_evidence': {slot.evidence_id: {
                         'document_id': slot.evidence.document_id,
                         'source_uri': slot.evidence.source_uri,
                         'source_pages': list(slot.evidence.source_pages),
                         'image_uri': slot.evidence.image_uri}
                         for slot in sealed.slots}})
            try:
                raw = self.generator.generate(query, requirements, sealed)
            except ModelOutputError:
                raise
            except Exception:
                return self._result(request, "failed", rounds, "GENERATOR_UNAVAILABLE",
                                    "答案模型暂时不可用。", route1=route1,
                                    route2=route2, degraded=degraded)
            structured = self.guard.check(raw, sealed)
            answer, citations = self.renderer.render(structured, sealed)
            status = "abstained" if structured.abstain else "answered"
            return self._result(request, status, rounds,
                                "MODEL_ABSTAINED" if structured.abstain else "ANSWERED",
                                answer, citations, route1, route2, degraded)
        except ModelOutputError:
            return self._result(request, "failed", rounds, "MODEL_OUTPUT_OR_CITATION_INVALID",
                                "答案格式或引用未通过校验。", route1=route1,
                                route2=route2, degraded=degraded)
        except (IntegrityError, ValueError):
            # Explicitly failed, not a fake knowledge-based 'no answer'.
            return self._result(request, "failed", rounds, "INTEGRITY_OR_CONTROL_FAILURE",
                                "知识库完整性或执行控制校验失败。", route1=route1,
                                route2=route2, degraded=degraded)
