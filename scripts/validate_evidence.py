#!/usr/bin/env python3
"""Validate a migrated canonical JSONL evidence store before index release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_rag.contracts import Evidence, IndexManifest, RequestContext
from controlled_rag.retrieval import assert_valid_evidence


def validate(manifest_path: Path, evidence_path: Path) -> int:
    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf8"))
    # This validation assumes the data owner has authorization for the entire KB.
    # It checks public/statically declared ACL labels, not real user auth.
    collected = []
    for row in evidence_path.read_text(encoding="utf8").splitlines():
        if not row.strip():
            continue
        payload = json.loads(row)
        evidence = Evidence.model_validate({
            **payload, "source_pages": tuple(payload["source_pages"]),
            "access_labels": frozenset(payload.get("access_labels", []))})
        collected.append(evidence)
    all_labels = frozenset(label for e in collected for label in e.access_labels)
    ctx = RequestContext(
        request_id="validation", knowledge_base_id=manifest.knowledge_base_id,
        index_version=manifest.index_version, principal_id="validation-operator",
        allowed_kb_ids=frozenset({manifest.knowledge_base_id}),
        access_labels=all_labels,
    )
    keys = set()
    for evidence in collected:
        if evidence.evidence_key in keys:
            raise ValueError(f"duplicate canonical evidence ID: {evidence.evidence_key}")
        keys.add(evidence.evidence_key)
        assert_valid_evidence(evidence, ctx, manifest)
    if not collected:
        raise ValueError("empty evidence store")
    print(f"Validated canonical evidence records: {len(collected)} (not a retrieval or model benchmark)")
    return len(collected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    args = ap.parse_args()
    validate(args.manifest, args.evidence)


if __name__ == "__main__":
    main()
