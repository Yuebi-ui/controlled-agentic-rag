"""Deterministic citation ID/metadata validity and separately annotated support."""
from __future__ import annotations

from .metrics import divide
from .schemas import CitationRecord


def evaluate_citation(rows: list[CitationRecord]) -> dict:
    if not rows:
        raise ValueError('empty citation dataset')
    total = valid = annotated = supported = 0
    gold_count = retrieved_gold = 0
    for row in rows:
        used = set()
        for ref in row.citations:
            total += 1
            canonical = row.canonical_evidence.get(ref.evidence_id)
            is_valid = (ref.evidence_id in row.allowed_evidence_ids and canonical is not None and
                        canonical.document_id == ref.document_id and
                        canonical.source_uri == ref.source_uri and
                        canonical.source_pages == ref.source_pages and
                        canonical.image_uri == ref.image_uri)
            valid += bool(is_valid)
            used.add(ref.evidence_id)
            if ref.semantic_supported is not None:
                annotated += 1
                supported += bool(ref.semantic_supported)
        if row.gold_required_evidence_ids is not None:
            gold_count += len(row.gold_required_evidence_ids)
            retrieved_gold += len(set(row.gold_required_evidence_ids) & used)
    return {
        'sample_count': len(rows), 'citation_count': total,
        'valid_citation_count': valid, 'citation_validity_rate': divide(valid, total),
        'semantic_support_annotated_count': annotated,
        'semantic_support_rate_annotated': divide(supported, annotated),
        'gold_required_citation_count': gold_count,
        'gold_evidence_citation_recall_micro': divide(retrieved_gold, gold_count),
        'semantic_support_requires_human_or_independent_verification': True,
    }
