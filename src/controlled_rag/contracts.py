"""Request-pinned data contracts for the Controlled Agentic RAG core.

Only the canonical store supplies evidence content and source metadata. Retriever
hits and LLM responses are never authoritative for those fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


Id = str
RouteState = Literal["sufficient", "partial", "failed"]


class IndexManifest(StrictModel):
    knowledge_base_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    document_versions: dict[str, str]


class RequestContext(StrictModel):
    request_id: str = Field(min_length=1)
    knowledge_base_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    allowed_kb_ids: frozenset[str]
    access_labels: frozenset[str] = frozenset()


class RetrievalHit(StrictModel):
    """Untrusted *index* hit. Content must be expanded from the canonical store."""

    candidate_id: str = Field(min_length=1)
    kind: Literal["text_child", "image"]
    parent_id: str | None = None
    knowledge_base_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_text_parent(self):
        if self.kind == "text_child" and not self.parent_id:
            raise ValueError("text_child must have a parent_id")
        return self


class Evidence(StrictModel):
    """Versioned canonical Parent or Image Evidence (not a Child document)."""

    evidence_key: str = Field(min_length=1)  # parent_id or image_id
    evidence_group_id: str = Field(min_length=1)
    kind: Literal["text_parent", "image"]
    text_surrogate: str = Field(min_length=1)
    knowledge_base_id: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_pages: tuple[int, ...] = Field(min_length=1)
    access_labels: frozenset[str] = frozenset()
    image_uri: str | None = None

    @model_validator(mode="after")
    def validate_source_pages(self):
        if any(p < 1 for p in self.source_pages) or len(set(self.source_pages)) != len(self.source_pages):
            raise ValueError("source_pages must be unique positive integers")
        if self.kind == "image" and not self.image_uri:
            raise ValueError("image evidence must have image_uri")
        return self


class AnswerBlock(StrictModel):
    heading: str | None = Field(default=None, max_length=200)
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_ids(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence_id within answer block")
        return self


class StructuredAnswer(StrictModel):
    abstain: bool
    abstain_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    answer_blocks: list[AnswerBlock]

    @model_validator(mode="after")
    def validate_state(self):
        if self.abstain:
            if self.answer_blocks or not self.abstain_reason:
                raise ValueError("abstaining must give a reason and zero answer_blocks")
        elif self.abstain_reason is not None or not self.answer_blocks:
            raise ValueError("answer requires blocks and no abstain_reason")
        return self


@dataclass(frozen=True)
class EvidenceSlot:
    """Request-scoped short id visible to the Answer Model."""

    evidence_id: str
    evidence: Evidence


@dataclass(frozen=True)
class FinalEvidence:
    request: RequestContext
    manifest: IndexManifest
    slots: tuple[EvidenceSlot, ...]

    def by_id(self) -> dict[str, Evidence]:
        return {slot.evidence_id: slot.evidence for slot in self.slots}


class Citation(StrictModel):
    evidence_id: str
    document_id: str
    source_uri: str
    source_pages: list[int]
    image_uri: str | None = None


class PipelineResult(StrictModel):
    request_id: str
    index_version: str
    final_status: Literal["answered", "abstained", "failed"]
    answer: str
    citations: list[Citation]
    retrieval_rounds: int = Field(ge=0, le=2)
    route1: RouteState | None = None
    route2: RouteState | None = None
    reason_code: str
    degraded_branches: list[str] = Field(default_factory=list)


class IntegrityError(RuntimeError):
    """A version, metadata, index or security boundary was violated."""


class ModelOutputError(RuntimeError):
    """Model output failed Pydantic or deterministic citation checks."""
