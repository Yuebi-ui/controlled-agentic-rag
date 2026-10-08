"""Fail-closed structured parsing, citation guard and metadata-only rendering."""

from __future__ import annotations

import re

from pydantic import ValidationError

from .contracts import (Citation, FinalEvidence, ModelOutputError,
                        StructuredAnswer)
from .retrieval import assert_valid_evidence


class CitationGuard:
    def check(self, raw: str, final: FinalEvidence) -> StructuredAnswer:
        try:
            answer = StructuredAnswer.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            raise ModelOutputError("invalid StructuredAnswer schema") from exc
        allowed = final.by_id()
        if not allowed:
            raise ModelOutputError("cannot validate model output without sealed evidence")
        for block in answer.answer_blocks:
            # Inline 'E9' textual citations are bypasses of the renderer contract.
            # Cite exclusively through the structured evidence_ids[] field.
            if re.search(r"(?<![\w])E\d+(?![\w])", block.text):
                raise ModelOutputError("inline evidence id in model-generated text")
            for evidence_id in block.evidence_ids:
                evidence = allowed.get(evidence_id)
                if evidence is None:
                    raise ModelOutputError("unknown or unsealed evidence id")
                assert_valid_evidence(evidence, final.request, final.manifest)
        return answer


class Renderer:
    """Never trusts model-provided page numbers, links, titles or document IDs."""

    def render(self, answer: StructuredAnswer, final: FinalEvidence) -> tuple[str, list[Citation]]:
        if answer.abstain:
            return answer.abstain_reason or "证据不足，无法回答。", []
        mapping = final.by_id()
        used: dict[str, Citation] = {}
        parts = []
        for block in answer.answer_blocks:
            suffix = " ".join(f"[{e_id}]" for e_id in block.evidence_ids)
            heading = f"{block.heading}\n" if block.heading else ""
            parts.append(f"{heading}{block.text} {suffix}")
            for evidence_id in block.evidence_ids:
                evidence = mapping[evidence_id]
                used[evidence_id] = Citation(
                    evidence_id=evidence_id,
                    document_id=evidence.document_id,
                    source_uri=evidence.source_uri,
                    source_pages=list(evidence.source_pages),
                    image_uri=evidence.image_uri,
                )
        return "\n\n".join(parts), list(used.values())
