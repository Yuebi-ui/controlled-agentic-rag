"""Unweighted RRF -> exact-id dedup -> Parent collapse -> canonical expansion."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import Evidence, IndexManifest, IntegrityError, RequestContext, RetrievalHit


class RankedRetriever(Protocol):
    def retrieve(self, query: str, request: RequestContext, top_k: int) -> Sequence[RetrievalHit]: ...


class CanonicalStore(Protocol):
    def fetch_many(self, evidence_keys: Sequence[str], request: RequestContext) -> Sequence[Evidence]: ...


@dataclass(frozen=True)
class RetrievalBatch:
    evidences: tuple[Evidence, ...]
    degraded_branches: tuple[str, ...]
    # Number of responding branches includes branches that returned zero hits.
    succeeded_branches: int


def unweighted_rrf(lists: dict[str, Sequence[RetrievalHit]], k: int = 60) -> list[RetrievalHit]:
    """Fuse branch ranks, detect cross-branch ID inconsistencies.

    Duplicate hit IDs in one branch count once. Branch weights are all 1.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    weights: dict[str, float] = defaultdict(float)
    exemplar: dict[str, RetrievalHit] = {}
    for branch, hits in lists.items():
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if hit.candidate_id in seen:
                continue
            seen.add(hit.candidate_id)
            previous = exemplar.setdefault(hit.candidate_id, hit)
            if previous != hit:
                raise IntegrityError(f"inconsistent retrieval metadata for {hit.candidate_id}")
            weights[hit.candidate_id] += 1.0 / (k + rank)
    return [exemplar[key] for key in sorted(weights, key=lambda x: (-weights[x], x))]


def collapse_to_evidence_keys(hits: Sequence[RetrievalHit]) -> list[str]:
    """Child siblings collapse to a unique Parent; image keys remain independent."""
    seen: set[str] = set()
    keys = []
    for hit in hits:
        key = hit.parent_id if hit.kind == "text_child" else hit.candidate_id
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def group_dedup(evidences: Sequence[Evidence], cap: int) -> list[Evidence]:
    if cap < 1:
        raise ValueError("cap must be positive")
    results: list[Evidence] = []
    groups: set[str] = set()
    for e in evidences:
        if e.evidence_group_id not in groups:
            groups.add(e.evidence_group_id)
            results.append(e)
        if len(results) == cap:
            break
    return results


def assert_valid_evidence(e: Evidence, request: RequestContext, manifest: IndexManifest) -> None:
    if request.knowledge_base_id not in request.allowed_kb_ids:
        raise IntegrityError("unauthorized knowledge base")
    if (manifest.index_version != request.index_version or
            manifest.knowledge_base_id != request.knowledge_base_id):
        raise IntegrityError("request/manifest version mismatch")
    if e.index_version != request.index_version or e.knowledge_base_id != request.knowledge_base_id:
        raise IntegrityError("evidence version or KB mismatch")
    if manifest.document_versions.get(e.document_id) != e.document_version:
        raise IntegrityError("document/manifest version mismatch or missing document")
    if not e.access_labels.issubset(request.access_labels):
        raise IntegrityError("request principal lacks evidence permission")


class RetrievalEngine:
    """Three optional branches at construction; at least 2 must be operational.

    Branch errors degrade from 3 to 2; no permission or integrity error can be
    silently treated as a recoverable branch failure.
    """

    def __init__(self, branches: dict[str, RankedRetriever], store: CanonicalStore,
                 manifest: IndexManifest, top_k: int = 10, pool_cap: int = 20):
        if set(branches) != {"bm25", "dense", "clip"}:
            raise ValueError("expected exactly bm25/dense/clip branches")
        self.branches = branches
        self.store = store
        self.manifest = manifest
        self.top_k = top_k
        self.pool_cap = pool_cap

    def retrieve(self, query: str, request: RequestContext) -> RetrievalBatch:
        if (self.manifest.index_version != request.index_version or
                self.manifest.knowledge_base_id != request.knowledge_base_id or
                request.knowledge_base_id not in request.allowed_kb_ids):
            raise IntegrityError("request not authorized for pinned index manifest")
        results: dict[str, Sequence[RetrievalHit]] = {}
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {name: executor.submit(retriever.retrieve, query, request, self.top_k)
                       for name, retriever in self.branches.items()}
            for name, future in futures.items():
                try:
                    result = list(future.result())
                    for hit in result:
                        if (hit.index_version != request.index_version or
                                hit.knowledge_base_id != request.knowledge_base_id):
                            raise IntegrityError("retrieval hit has stale index/KB version")
                    results[name] = result
                except IntegrityError:
                    raise
                except ValueError as exc:
                    raise IntegrityError("invalid retrieval branch hit schema") from exc
                except Exception:
                    failed.append(name)
        if len(results) < 2:
            raise IntegrityError("less than two retrieval branches available")

        hits = unweighted_rrf(results)
        expected_kinds = {}
        for hit in hits:
            key = hit.parent_id if hit.kind == "text_child" else hit.candidate_id
            kind = "text_parent" if hit.kind == "text_child" else "image"
            if key in expected_kinds and expected_kinds[key] != kind:
                raise IntegrityError("Parent/Image evidence key collision")
            expected_kinds[key] = kind
        keys = collapse_to_evidence_keys(hits)[:self.pool_cap]
        if not keys:
            return RetrievalBatch((), tuple(failed), len(results))
        try:
            expanded = list(self.store.fetch_many(keys, request))
        except IntegrityError:
            raise
        except Exception as exc:
            raise IntegrityError("canonical store batch fetch failed") from exc
        mapping = {e.evidence_key: e for e in expanded}
        if len(mapping) != len(expanded) or set(keys) != set(mapping):
            raise IntegrityError("missing/duplicate/unexpected canonical evidence")
        ordered = []
        for key in keys:
            e = mapping[key]
            if e.kind != expected_kinds[key]:
                raise IntegrityError("canonical evidence kind mismatch")
            assert_valid_evidence(e, request, self.manifest)
            ordered.append(e)
        return RetrievalBatch(tuple(group_dedup(ordered, self.pool_cap)), tuple(failed), len(results))
