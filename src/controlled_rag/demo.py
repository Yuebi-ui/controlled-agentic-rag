"""Fully deterministic, SYNTHETIC offline control-flow demonstration.

No trained model, Tesla PDF, MongoDB, Milvus or real metrics are used here.
"""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Sequence

from .contracts import Evidence, IndexManifest, RequestContext, RetrievalHit
from .orchestrator import ControlledRAGPipeline
from .retrieval import RetrievalEngine
from .routing import EvidenceRouter


class DemoRetriever:
    def __init__(self, branch: str):
        self.branch = branch

    def retrieve(self, query: str, request: RequestContext, top_k: int):
        ids = []
        if "儿童锁" in query:
            ids.append(("child-lock-on", "parent-lock-on"))
            if "补查缺失需求" in query or "整体补查" in query:
                ids.append(("child-lock-scope", "parent-lock-scope"))
        if "空调" in query:
            ids.append(("child-ac", "parent-ac"))
        if self.branch == "clip":
            ids = []  # The synthetic example contains no image evidence.
        if self.branch == "dense":
            ids = list(reversed(ids))
        return [
            RetrievalHit(candidate_id=child, parent_id=parent, kind="text_child",
                         knowledge_base_id=request.knowledge_base_id,
                         index_version=request.index_version)
            for child, parent in ids[:top_k]
        ]


class DemoStore:
    def __init__(self, corpus: list[Evidence]):
        self.items = {e.evidence_key: e for e in corpus}

    def fetch_many(self, evidence_keys, request):
        return [self.items[k] for k in evidence_keys if k in self.items]


class DemoScorer:
    def score_many(self, query: str, evidences: Sequence[Evidence]) -> list[float]:
        def score(e: Evidence):
            if ("开启" in query or "打开" in query) and "影响" not in query:
                return .96 if e.evidence_key == "parent-lock-on" else .09
            if "影响" in query and "开启" not in query and "打开" not in query:
                return .97 if e.evidence_key == "parent-lock-scope" else .08
            if "儿童锁" in query:
                return .90 if e.evidence_key == "parent-lock-on" else (.88 if e.evidence_key == "parent-lock-scope" else .05)
            return .92 if e.evidence_key == "parent-ac" and "空调" in query else .03
        return [score(e) for e in evidences]


class DemoLabelModel:
    def predict(self, features):
        if features[4] == 1.0:
            return "sufficient"
        if features[4] == 0.0:
            return "failed"
        return "partial"


class DemoDecomposer:
    def decompose(self, original_query: str):
        if "儿童锁" in original_query and "影响" in original_query and ("开启" in original_query or "打开" in original_query):
            return ["如何开启儿童锁", "儿童锁影响哪些车门"]
        return [original_query]


class DemoRewriter:
    def rewrite_missing(self, query, missing_requirements):
        return "补查缺失需求：" + "；".join(missing_requirements)

    def rewrite_failed(self, query):
        return "整体补查：" + query


class DemoGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, original_query, requirements, final):
        self.calls += 1
        blocks = [
            {"heading": None, "text": slot.evidence.text_surrogate,
             "evidence_ids": [slot.evidence_id]}
            for slot in final.slots
        ]
        return json.dumps({"abstain": False, "abstain_reason": None,
                           "answer_blocks": blocks}, ensure_ascii=False)


def demo_corpus() -> list[Evidence]:
    """Invented example facts, not factual Tesla instructions."""
    def doc(key, group, text, page):
        return Evidence(
            evidence_key=key, evidence_group_id=group, kind="text_parent",
            text_surrogate=text, knowledge_base_id="synthetic-manual",
            index_version="demo-v1", document_id="demo-pdf",
            document_version="demo-doc-v1", source_uri="demo://fictional-manual.pdf",
            source_pages=(page,),
        )
    return [
        doc("parent-lock-on", "g-on", "【虚构演示手册】儿童锁通过控制菜单中的儿童锁选项开启。", 1),
        doc("parent-lock-scope", "g-scope", "【虚构演示手册】儿童锁作用于两侧后车门。", 2),
        doc("parent-ac", "g-ac", "【虚构演示手册】空调可以在控制菜单中设置。", 3),
    ]


def build_demo_pipeline(generator=None) -> ControlledRAGPipeline:
    manifest = IndexManifest(knowledge_base_id="synthetic-manual",
                             index_version="demo-v1",
                             document_versions={"demo-pdf": "demo-doc-v1"})
    retrieval = RetrievalEngine(
        {name: DemoRetriever(name) for name in ("bm25", "dense", "clip")},
        DemoStore(demo_corpus()), manifest,
    )
    scorer = DemoScorer()
    router = EvidenceRouter(scorer, DemoLabelModel(), coverage_threshold=.50)
    return ControlledRAGPipeline(
        retrieval, scorer, router, DemoDecomposer(), DemoRewriter(),
        generator or DemoGenerator(),
    )


def demo_request(request_id=None) -> RequestContext:
    return RequestContext(
        request_id=request_id or uuid.uuid4().hex,
        knowledge_base_id="synthetic-manual", index_version="demo-v1",
        principal_id="demo-user", allowed_kb_ids=frozenset({"synthetic-manual"}),
    )


def main():
    parser = argparse.ArgumentParser(description="Synthetic Controlled RAG demonstration")
    parser.add_argument("--query", default="如何开启儿童锁，以及儿童锁影响哪些车门？")
    args = parser.parse_args()
    print(json.dumps(build_demo_pipeline().run(args.query, demo_request()).model_dump(),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
