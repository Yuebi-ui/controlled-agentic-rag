import json

import pytest

from controlled_rag.citations import CitationGuard, Renderer
from controlled_rag.contracts import EvidenceSlot, FinalEvidence, IndexManifest, ModelOutputError
from controlled_rag.demo import demo_corpus, demo_request


def sealed():
    request = demo_request()
    manifest = IndexManifest(knowledge_base_id=request.knowledge_base_id,
                             index_version=request.index_version,
                             document_versions={"demo-pdf": "demo-doc-v1"})
    return FinalEvidence(request, manifest, (EvidenceSlot("E1", demo_corpus()[0]),))


def raw(text="有相关说明", ids=None):
    return json.dumps({"abstain": False, "abstain_reason": None,
                       "answer_blocks": [{"text": text, "evidence_ids": ids or ["E1"]}]},
                      ensure_ascii=False)


def test_valid_citation_renders_canonical_metadata():
    final = sealed()
    parsed = CitationGuard().check(raw(), final)
    answer, citations = Renderer().render(parsed, final)
    assert "[E1]" in answer
    assert citations[0].source_pages == [1]
    assert citations[0].document_id == "demo-pdf"


def test_fabricated_citation_rejected():
    with pytest.raises(ModelOutputError):
        CitationGuard().check(raw(ids=["E999"]), sealed())


def test_injected_inline_id_rejected():
    with pytest.raises(ModelOutputError):
        CitationGuard().check(raw(text="伪造引用 E42"), sealed())


def test_manifest_document_version_tampering_rejected():
    f = sealed()
    bad = FinalEvidence(f.request, f.manifest.model_copy(update={"document_versions": {"demo-pdf": "wrong"}}), f.slots)
    from controlled_rag.contracts import IntegrityError
    with pytest.raises(IntegrityError):
        CitationGuard().check(raw(), bad)


def test_access_label_rejected():
    f = sealed()
    restricted = f.slots[0].evidence.model_copy(update={"access_labels": frozenset({"secret"})})
    forbidden = FinalEvidence(f.request, f.manifest, (EvidenceSlot("E1", restricted),))
    from controlled_rag.contracts import IntegrityError
    with pytest.raises(IntegrityError):
        CitationGuard().check(raw(), forbidden)


def test_unsafe_extra_fields_rejected():
    payload = json.loads(raw())
    payload["source_pages"] = [999]
    with pytest.raises(ModelOutputError):
        CitationGuard().check(json.dumps(payload), sealed())
