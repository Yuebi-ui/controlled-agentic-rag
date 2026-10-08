"""Generation-only answer contract; does not replace the downstream Citation Guard."""

from __future__ import annotations

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_EVIDENCE_ID_PATTERN = re.compile(r"E[1-9][0-9]*\Z")


class StrictAnswerModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class AnswerBlock(StrictAnswerModel):
    heading: str | None = Field(default=None, max_length=200)
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not _EVIDENCE_ID_PATTERN.fullmatch(item) for item in value):
            raise ValueError("evidence_ids must be request-scoped IDs such as E1")
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must be unique within each block")
        return value


class StructuredAnswer(StrictAnswerModel):
    abstain: bool
    abstain_reason: str | None
    answer_blocks: list[AnswerBlock]

    @model_validator(mode="after")
    def validate_abstention_state(self) -> "StructuredAnswer":
        if self.abstain:
            if not self.abstain_reason or not self.abstain_reason.strip():
                raise ValueError("abstain=true requires abstain_reason")
            if self.answer_blocks:
                raise ValueError("abstain=true requires empty answer_blocks")
        else:
            if self.abstain_reason is not None:
                raise ValueError("abstain=false requires abstain_reason=null")
            if not self.answer_blocks:
                raise ValueError("abstain=false requires nonempty answer_blocks")
        return self


__all__ = ["AnswerBlock", "StructuredAnswer"]
