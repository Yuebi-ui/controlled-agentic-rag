"""Versioned JSONL contracts for offline measurements and later real runs."""
from __future__ import annotations

import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Route = Literal['sufficient', 'partial', 'failed']
Status = Literal['answered', 'abstained', 'failed']


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    id: str = Field(min_length=1)


def _distinct(xs: list[str], name: str) -> None:
    if len(xs) != len(set(xs)) or any(not x.strip() for x in xs):
        raise ValueError(f'{name} requires distinct nonempty IDs')


class RetrievalRecord(StrictRecord):
    gold_parent_ids: list[str] = Field(default_factory=list)
    ranked_parent_ids: list[str] = Field(default_factory=list)
    gold_image_ids: list[str] = Field(default_factory=list)
    ranked_image_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check(self):
        for field in ('gold_parent_ids', 'ranked_parent_ids', 'gold_image_ids', 'ranked_image_ids'):
            _distinct(getattr(self, field), field)
        if not self.gold_parent_ids and not self.gold_image_ids:
            raise ValueError('at least one gold Parent or Image ID is required')
        return self


class RerankerRecord(StrictRecord):
    positive_ids: list[str] = Field(min_length=1)
    before: list[str]
    after: list[str]
    after_scores: dict[str, float] | None = None

    @model_validator(mode='after')
    def check(self):
        for field in ('positive_ids', 'before', 'after'):
            _distinct(getattr(self, field), field)
        if set(self.before) != set(self.after):
            raise ValueError('reranking cannot change the candidate pool')
        if self.after_scores is not None and set(self.after_scores) != set(self.after):
            raise ValueError('after_scores must score each candidate exactly once')
        return self


class RouterRecord(StrictRecord):
    gold_state: Route
    predicted_state: Route
    raw_model_state: Route | None = None
    probabilities: dict[Route, float] | None = None

    @model_validator(mode='after')
    def check(self):
        if self.probabilities is not None:
            if set(self.probabilities) != {'sufficient', 'partial', 'failed'}:
                raise ValueError('all three class probabilities required')
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in self.probabilities.values()):
                raise ValueError('probability must be finite in [0,1]')
            if abs(sum(self.probabilities.values()) - 1) > 1e-6:
                raise ValueError('class probabilities must sum to 1')
        return self


class Round2Record(StrictRecord):
    round1_state: Route | None
    round2_state: Route | None = None
    retrieval_rounds: int = Field(ge=0, le=2)
    final_status: Status
    gold_answerable: bool
    requirement_count: int = Field(ge=1)
    round1_covered: int = Field(ge=0)
    final_covered: int = Field(ge=0)
    mandatory_round1_keys: list[str] = Field(default_factory=list)
    merged_evidence_keys: list[str] | None = None
    final_evidence_keys: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check(self):
        if self.round1_covered > self.requirement_count or self.final_covered > self.requirement_count:
            raise ValueError('covered requirements cannot exceed total')
        if self.retrieval_rounds == 0 and self.round1_state is not None:
            raise ValueError('no retrieval implies no first router decision')
        if self.retrieval_rounds >= 1 and self.round1_state is None:
            raise ValueError('first router decision required')
        if (self.round2_state is None) != (self.retrieval_rounds != 2):
            raise ValueError('round2_state required only for two-round executions')
        for field in ('mandatory_round1_keys', 'merged_evidence_keys', 'final_evidence_keys'):
            value = getattr(self, field)
            if value is not None:
                _distinct(value, field)
        return self


class GenerationRecord(StrictRecord):
    model_variant: Literal['base', 'sft', 'dpo']
    gold_answerable: bool
    schema_valid: bool
    predicted_abstain: bool | None = None
    correctness: bool | None = None  # independently annotated, answered cases only
    faithfulness: bool | None = None  # independently annotated, answered cases only

    @model_validator(mode='after')
    def check(self):
        if self.schema_valid and self.predicted_abstain is None:
            raise ValueError('valid schema requires predicted_abstain')
        if not self.schema_valid and self.predicted_abstain is not None:
            raise ValueError('unparseable output cannot have verified abstain state')
        if self.predicted_abstain is not False and (self.correctness is not None or self.faithfulness is not None):
            raise ValueError('answer correctness/faithfulness labels apply only to generated answers')
        return self


class CanonicalCitation(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    document_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_pages: list[int] = Field(min_length=1)
    image_uri: str | None = None

    @model_validator(mode='after')
    def check(self):
        if any(x < 1 for x in self.source_pages) or len(set(self.source_pages)) != len(self.source_pages):
            raise ValueError('source pages must be unique and positive')
        return self


class CitationObservation(CanonicalCitation):
    evidence_id: str = Field(min_length=1)
    semantic_supported: bool | None = None  # externally reviewed, never auto-inferred


class CitationRecord(StrictRecord):
    allowed_evidence_ids: list[str]
    canonical_evidence: dict[str, CanonicalCitation]
    citations: list[CitationObservation]
    gold_required_evidence_ids: list[str] | None = None

    @model_validator(mode='after')
    def check(self):
        _distinct(self.allowed_evidence_ids, 'allowed_evidence_ids')
        if set(self.canonical_evidence) != set(self.allowed_evidence_ids):
            raise ValueError('canonical_evidence must list each allowed ID')
        if self.gold_required_evidence_ids is not None:
            _distinct(self.gold_required_evidence_ids, 'gold_required_evidence_ids')
        return self


class SafetyRecord(StrictRecord):
    attack_type: str = Field(min_length=1)
    expected_status: Status
    actual_status: Status
    expected_reason_codes: list[str] = Field(default_factory=list)
    actual_reason_code: str = Field(min_length=1)
    leaked_unauthorized: bool = False
    must_not_generate: bool = False
    generation_called: bool = False
    attack_succeeded: bool | None = None  # reviewed attack objective, not inferred from HTTP status

    @model_validator(mode='after')
    def check(self):
        if self.attack_type == 'prompt_injection' and self.attack_succeeded is None:
            raise ValueError('prompt injection requires an independently reviewed attack outcome')
        return self


class BenchmarkRecord(StrictRecord):
    scenario: str = Field(min_length=1)
    start_ms: float
    end_ms: float
    http_status: int = Field(ge=100, le=599)
    concurrency: int = Field(ge=1)
    gpu_memory_mb: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def check(self):
        if self.end_ms < self.start_ms:
            raise ValueError('end_ms must not precede start_ms')
        return self


class AblationRecord(StrictRecord):
    variant: str = Field(min_length=1)
    metrics: dict[str, float] = Field(min_length=1)


class BadCaseRecord(StrictRecord):
    category: str = Field(min_length=1)
    severity: Literal['critical', 'high', 'medium', 'low']
    expected_status: Status
    actual_status: Status
    expected_reason_codes: list[str] = Field(default_factory=list)
    actual_reason_code: str = Field(min_length=1)
    actual_retrieval_rounds: int = Field(ge=0)
    max_allowed_rounds: int = Field(default=2, ge=0, le=2)
    forbidden_citation_ids: list[str] = Field(default_factory=list)
    observed_citation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check(self):
        for field in ('forbidden_citation_ids', 'observed_citation_ids'):
            _distinct(getattr(self, field), field)
        return self


class WorkflowRecord(StrictRecord):
    """One question's trace-derived decisions plus independently specified gold labels.

    `gold_answerable` is corpus-level answerability, not an automatic judgment of
    whether the variant retrieved enough evidence. Route1 labels should be reviewed
    against that variant's actual first-round evidence in a real study.
    """
    gold_answerable: bool
    gold_round1_state: Route
    predicted_round1_state: Route | None = None
    expected_round2: bool
    actual_round2: bool
    retrieval_rounds: int = Field(ge=0, le=2)
    final_gate_state: Route | None = None
    final_status: Status
    reason_code: str = Field(min_length=1)
    sealed_final_present: bool
    mandatory_round1_keys: list[str] = Field(default_factory=list)
    merged_evidence_keys: list[str] | None = None
    elapsed_ms: float = Field(ge=0)

    @model_validator(mode='after')
    def validate_trace(self):
        _distinct(self.mandatory_round1_keys, 'mandatory_round1_keys')
        if self.merged_evidence_keys is not None:
            _distinct(self.merged_evidence_keys, 'merged_evidence_keys')
        # Do not silently accept an internally contradictory trace as evidence.
        if self.actual_round2 != (self.retrieval_rounds == 2):
            raise ValueError('actual_round2 does not match retrieval_rounds')
        if self.gold_round1_state == 'sufficient' and self.expected_round2:
            raise ValueError('sufficient gold round1 evidence should not trigger Round2')
        return self
