import pytest

from controlled_rag.contracts import Evidence, IntegrityError, RetrievalHit
from controlled_rag.demo import DemoRetriever, DemoStore, demo_corpus, demo_request
from controlled_rag.retrieval import (RetrievalEngine, collapse_to_evidence_keys,
                                      group_dedup, unweighted_rrf)


def hit(child, parent, kind="text_child"):
    return RetrievalHit(candidate_id=child, parent_id=parent if kind == "text_child" else None,
                        kind=kind, knowledge_base_id="synthetic-manual", index_version="demo-v1")


def test_rrf_unweighted_dedup_exact_id():
    ranks = {
        "bm25": [hit("c1", "p1"), hit("c2", "p1")],
        "dense": [hit("c1", "p1"), hit("c3", "p2")],
        "clip": [],
    }
    fused = unweighted_rrf(ranks)
    assert fused[0].candidate_id == "c1"
    assert collapse_to_evidence_keys(fused) == ["p1", "p2"]


def test_rrf_id_collision_fail_closed():
    with pytest.raises(IntegrityError):
        unweighted_rrf({"a": [hit("c1", "p1")], "b": [hit("c1", "p2")]})


def test_parent_dedup():
    data = demo_corpus()
    assert [x.evidence_key for x in group_dedup([data[0], data[0], data[1]], cap=2)] == [
        "parent-lock-on", "parent-lock-scope"]


def test_request_version_mismatch_fails_before_retrieval():
    from controlled_rag.demo import build_demo_pipeline
    pipeline = build_demo_pipeline()
    req = demo_request().model_copy(update={"index_version": "other-version"})
    result = pipeline.run("如何开启儿童锁", req)
    assert result.final_status == "failed"
    assert result.retrieval_rounds == 0


def test_unauthorized_request_fails_before_retrieval():
    from controlled_rag.demo import build_demo_pipeline
    req = demo_request().model_copy(update={"allowed_kb_ids": frozenset()})
    result = build_demo_pipeline().run("如何开启儿童锁", req)
    assert result.final_status == "failed"


def test_two_branch_failure_fail_closed():
    from controlled_rag.demo import build_demo_pipeline
    pipeline = build_demo_pipeline()

    class DeadRetriever:
        def retrieve(self, *args, **kwargs):
            raise OSError("temporarily unavailable")

    pipeline.retrieval.branches["bm25"] = DeadRetriever()
    pipeline.retrieval.branches["clip"] = DeadRetriever()
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.reason_code == "INTEGRITY_OR_CONTROL_FAILURE"


def test_one_branch_failure_degraded_continue():
    from controlled_rag.demo import build_demo_pipeline
    pipeline = build_demo_pipeline()

    class DeadRetriever:
        def retrieve(self, *args, **kwargs):
            raise OSError("temporarily unavailable")

    pipeline.retrieval.branches["clip"] = DeadRetriever()
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert result.final_status == "answered"
    assert result.degraded_branches == ["clip"]


def test_image_key_collision_with_text_parent_fails_closed():
    from controlled_rag.demo import build_demo_pipeline
    pipeline = build_demo_pipeline()

    class CollidingImages:
        def retrieve(self, query, request, top_k):
            return [hit("parent-lock-on", None, kind="image")]

    pipeline.retrieval.branches["clip"] = CollidingImages()
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"


def test_backend_error_does_not_turn_into_answer():
    from controlled_rag.demo import build_demo_pipeline
    pipeline = build_demo_pipeline()

    class DeadStore:
        def fetch_many(self, keys, request):
            raise OSError("Mongo timeout")

    pipeline.retrieval.store = DeadStore()
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.citations == []


def test_branch_deadline_degrades_one_stalled_retriever():
    import time
    from controlled_rag.demo import build_demo_pipeline

    class SlowRetriever:
        def retrieve(self, *args, **kwargs):
            time.sleep(0.12)
            return []

    pipeline = build_demo_pipeline()
    pipeline.retrieval.branch_timeout_s = 0.025
    pipeline.retrieval.branches["clip"] = SlowRetriever()
    began = time.monotonic()
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert (time.monotonic() - began) < 0.11
    assert result.final_status == "answered"
    assert "clip" in result.degraded_branches


def test_branch_deadline_two_stalled_branches_fail_closed():
    import time
    from controlled_rag.demo import build_demo_pipeline

    class SlowRetriever:
        def retrieve(self, *args, **kwargs):
            time.sleep(0.10)
            return []

    pipeline = build_demo_pipeline()
    pipeline.retrieval.branch_timeout_s = 0.025
    pipeline.retrieval.branches["bm25"] = SlowRetriever()
    pipeline.retrieval.branches["clip"] = SlowRetriever()
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.reason_code == "INTEGRITY_OR_CONTROL_FAILURE"


def test_invalid_branch_deadline_rejected():
    from controlled_rag.demo import build_demo_pipeline
    from controlled_rag.retrieval import RetrievalEngine
    demo = build_demo_pipeline().retrieval
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RetrievalEngine(demo.branches, demo.store, demo.manifest, branch_timeout_s=value)
