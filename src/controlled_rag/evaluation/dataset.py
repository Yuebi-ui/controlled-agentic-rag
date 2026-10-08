"""Prevent exact ID and document leakage between evaluation splits.

Input is a JSONL audit manifest with {id, split, document_id, text_hash?}.
The audit does not claim that semantic near-duplicates are detected.
"""
from __future__ import annotations

from collections import defaultdict


def audit_splits(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('empty dataset audit')
    ids: dict[str, set[str]] = defaultdict(set)
    documents: dict[str, set[str]] = defaultdict(set)
    content: dict[str, set[str]] = defaultdict(set)
    invalid = []
    seen_split_ids: set[tuple[str, str]] = set()
    duplicate_ids_within_split: set[str] = set()
    for idx, item in enumerate(rows, 1):
        for key in ('id', 'split', 'document_id'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                invalid.append({'row': idx, 'missing_or_invalid': key})
        if any(not isinstance(item.get(x), str) or not item[x].strip()
               for x in ('id', 'split', 'document_id')):
            continue
        split_key = (item['split'], item['id'])
        if split_key in seen_split_ids:
            duplicate_ids_within_split.add(item['id'])
        seen_split_ids.add(split_key)
        ids[item['id']].add(item['split'])
        documents[item['document_id']].add(item['split'])
        if item.get('text_hash'):
            content[str(item['text_hash'])].add(item['split'])
    leaking_ids = sorted(k for k, splits in ids.items() if len(splits) > 1)
    leaking_docs = sorted(k for k, splits in documents.items() if len(splits) > 1)
    leaking_hashes = sorted(k for k, splits in content.items() if len(splits) > 1)
    return {'sample_count': len(rows), 'invalid_rows': invalid,
            'duplicate_ids_within_split': sorted(duplicate_ids_within_split),
            'cross_split_id_leaks': leaking_ids, 'cross_split_document_leaks': leaking_docs,
            'cross_split_exact_hash_leaks': leaking_hashes,
            'passed': not (invalid or duplicate_ids_within_split or leaking_ids or leaking_docs or leaking_hashes)}
