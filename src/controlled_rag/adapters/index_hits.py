"""Fail-fast adapter for LangChain Documents produced by the old retrievers.

Do not infer parent IDs or index versions. The legacy Milvus hybrid method mixes
sparse/dense internally; a dedicated dense index query is needed for strict V4.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..contracts import IntegrityError, RequestContext, RetrievalHit


class LegacyDocumentRetrieverAdapter:
    def __init__(self, retriever: Any, kind: str = "text_child"):
        if kind not in ("text_child", "image"):
            raise ValueError("unsupported kind")
        self.retriever, self.kind = retriever, kind

    def retrieve(self, query: str, request: RequestContext, top_k: int) -> Sequence[RetrievalHit]:
        documents = self.retriever.retrieve_topk(query, top_k)
        hits = []
        for doc in documents:
            metadata = doc.metadata
            # Only migrated indices carry the canonical ID/version contract.
            key = "unique_id" if self.kind == "text_child" else "image_id"
            if not all(metadata.get(field) for field in (key, "index_version", "knowledge_base_id")):
                raise IntegrityError("legacy index lacks required IDs/versions; rebuild/migrate")
            if self.kind == "text_child" and not metadata.get("parent_id"):
                raise IntegrityError("legacy Child lacks parent_id; cannot Parent Collapse")
            hits.append(RetrievalHit(
                candidate_id=str(metadata[key]), kind=self.kind,
                parent_id=str(metadata["parent_id"]) if self.kind == "text_child" else None,
                knowledge_base_id=str(metadata["knowledge_base_id"]),
                index_version=str(metadata["index_version"]),
            ))
        return hits
