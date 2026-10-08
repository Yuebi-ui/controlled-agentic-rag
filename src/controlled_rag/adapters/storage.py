"""Canonical MongoDB adapter; requires new Parent/Image Evidence contract.

Old source records may lack parent_id/version/group metadata: migrate/validate
before serving. We intentionally do NOT invent missing metadata.
"""

from __future__ import annotations

from typing import Sequence

from ..contracts import Evidence, IntegrityError, RequestContext


class MongoCanonicalStore:
    def __init__(self, mongo_uri: str, db_name: str, collection: str = "evidence"):
        from pymongo import MongoClient
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection]

    def fetch_many(self, evidence_keys: Sequence[str], request: RequestContext) -> list[Evidence]:
        if not evidence_keys:
            return []
        projection = {"_id": 0}
        records = list(self.collection.find(
            {"evidence_key": {"$in": list(evidence_keys)},
             "index_version": request.index_version,
             "knowledge_base_id": request.knowledge_base_id}, projection))
        try:
            return [Evidence.model_validate({
                **record,
                "source_pages": tuple(record["source_pages"]),
                "access_labels": frozenset(record.get("access_labels", [])),
            }) for record in records]
        except (KeyError, ValueError) as exc:
            raise IntegrityError("canonical store record fails Evidence contract") from exc
