"""Bridge the standalone DPO/SFT adapter to the ControlledRAGPipeline protocol.

The production adapter is DPO-first. SFT is an explicitly selected evaluation
baseline; no automatic fallback takes place if the DPO endpoint is unavailable.
The sealed evidence is checked *before* being sent to the model endpoint.
"""
from __future__ import annotations

from typing import Sequence

from controlled_answer import (EvidenceItem, GenerationRequest, Qwen3AnswerAdapter,
                               GenerationContractError, create_answer_model)
from ..contracts import FinalEvidence, IntegrityError, ModelOutputError
from ..retrieval import assert_valid_evidence


class PipelineQwen3AnswerModel:
    def __init__(self, adapter: Qwen3AnswerAdapter | None = None):
        self.adapter = adapter if adapter is not None else create_answer_model()

    def generate(self, original_query: str, requirements: Sequence[str], final: FinalEvidence) -> str:
        if not final.slots:
            raise IntegrityError('sealed final evidence is empty')
        if (final.request.index_version != final.manifest.index_version or
                final.request.knowledge_base_id != final.manifest.knowledge_base_id):
            raise IntegrityError('final manifest/request version mismatch')
        if len({s.evidence_id for s in final.slots}) != len(final.slots):
            raise IntegrityError('repeated sealed evidence ID')
        for slot in final.slots:
            assert_valid_evidence(slot.evidence, final.request, final.manifest)
        request = GenerationRequest(
            original_query=original_query,
            atomic_requirements=tuple(requirements),
            final_evidence=tuple(EvidenceItem(s.evidence_id, s.evidence.text_surrogate) for s in final.slots),
        )
        # The core CitationGuard repeats independent parsing, allowed-ID, ACL,
        # version and provenance checks. Two independent Pydantic schemas are used.
        try:
            answer = self.adapter.generate(request)
        except GenerationContractError as exc:
            raise ModelOutputError('DPO/SFT generation schema or ID contract failure') from exc
        return answer.model_dump_json()
