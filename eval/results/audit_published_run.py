"""Audit maintainer-provided published records from anonymized labels.

This establishes byte-level integrity and metric consistency, not independent
validation of model execution, annotation origins or physical GPU telemetry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from controlled_rag.evaluation.e2e import summarize_end_to_end, summarize_outcome_ablation
from controlled_rag.evaluation.published_run import (
    summarize_benchmark,
    summarize_generation,
    summarize_retrieval,
    summarize_round2,
    summarize_router,
    summarize_safety,
)

def require(condition: bool, message: str = "audit assertion failed") -> None:
    """Invariant that cannot be compiled away with Python -O."""
    if not condition:
        raise AssertionError(message)


RUN_ID = "RUN-EXAMPLE-2026-001"
SOURCE_TYPE = "real_audited"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_equal(actual, expected, label="value"):
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None:
            require((actual is expected), (f"{label}: {actual!r} != {expected!r}"))
        else:
            require((math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)), (f"{label}: {actual!r} != {expected!r}"))
    elif isinstance(actual, dict) and isinstance(expected, dict):
        require((actual.keys() == expected.keys()), (f"{label}: dict keys differ"))
        for key in actual:
            assert_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        require((len(actual) == len(expected)), (f"{label}: list lengths differ"))
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_equal(a, b, f"{label}[{i}]")
    else:
        require((actual == expected), (f"{label}: {actual!r} != {expected!r}"))


def audit(root: Path) -> dict:
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    summary = read_json(root / "summary.json")
    require(manifest.get("dataset_manifest_sha256") == sha(root / "cohort_anonymized.jsonl"),
            "cohort SHA mismatch")
    require(manifest["evaluation"].get("ablation_per_query_outcome_sha256") == sha(root / "ablation_outcomes.jsonl"),
            "ablation SHA mismatch")
    require(manifest["evaluation"].get("e2e_per_query_outcome_sha256") == sha(root / "workflow_anonymized.jsonl"),
            "workflow SHA mismatch")

    require((manifest["run_id"] == RUN_ID == summary["run_id"]), ('audit assertion failed'))
    require((manifest["source_type"] == SOURCE_TYPE == summary["source_type"]), ('audit assertion failed'))
    require(summary["source_type_claimant"] == "project_maintainer", "missing source claim attribution")
    require(summary["provenance_independently_verified"] is False, "unproven public provenance")
    require("not independently verified" in summary["disclosure"], "ambiguous provenance disclosure")
    require((summary["real_system_verified"] is False), ('audit assertion failed'))
    require(manifest["source_type_claimant"] == "project_maintainer", "missing source attribution")
    require(manifest["provenance_independently_verified"] is False, "unverified independent audit")
    require("not verified" in manifest["verification_status"], "missing independent provenance disclaimer")
    require((manifest["hardware"]["physical_hardware_measured"] is False), ('audit assertion failed'))
    require((manifest["evaluation"]["independent_human_annotation_completed"] is False), ('audit assertion failed'))
    require((all(v is False for v in manifest["auditable_claims"].values())), ('audit assertion failed'))

    # Reject missing/extra entries, path traversal and symlink redirection.
    expected_files = {
        "RESULTS.md", "summary.json", "cohort_anonymized.jsonl",
        "retrieval_anonymized.jsonl", "router_anonymized.jsonl",
        "workflow_anonymized.jsonl", "generation_judgements.jsonl",
        "ablation_outcomes.jsonl", "safety_anonymized.jsonl",
        "benchmark_anonymized.jsonl",
    }
    require(set(manifest["files"]) == expected_files, "manifest file set mismatch")
    # Every manifest file is fixed by SHA-256 and byte length.
    for name, meta in manifest["files"].items():
        path = root / name
        require(not path.is_symlink(), f"symlink forbidden: {name}")
        require(path.is_file(), f"missing evidence: {name}")
        require((sha(path) == meta["sha256"]), (f"SHA mismatch: {name}"))
        require((path.stat().st_size == meta["bytes"]), (f"byte size mismatch: {name}"))

    # Hash labels mean something only if the referenced input exists.
    unverifiable = (
        "gold_annotation_protocol_sha256", "prompts_and_decoding_config_sha256",
        "query_split_sha256", "qwen3_dpo_checkpoint_sha256",
        "qwen3_sft_checkpoint_sha256", "reranker_checkpoint_sha256",
        "router_checkpoint_sha256",
    )
    require(all(manifest[key] is None for key in unverifiable),
            "the example must not invent hashes for absent model/config artifacts")

    cohort = read_jsonl(root / "cohort_anonymized.jsonl")
    ids_sha = hashlib.sha256(("\n".join(row["id"] for row in cohort) + "\n").encode("utf-8")).hexdigest()
    require(manifest["evaluation"]["frozen_query_ids_sha256"] == ids_sha,
            "query ID SHA mismatch")
    retrieval = read_jsonl(root / "retrieval_anonymized.jsonl")
    router = read_jsonl(root / "router_anonymized.jsonl")
    workflow = read_jsonl(root / "workflow_anonymized.jsonl")
    generation = read_jsonl(root / "generation_judgements.jsonl")
    ablation = read_jsonl(root / "ablation_outcomes.jsonl")
    safety = read_jsonl(root / "safety_anonymized.jsonl")
    benchmark = read_jsonl(root / "benchmark_anonymized.jsonl")

    require((len(cohort) == 300), ('audit assertion failed'))
    ids = {x["id"] for x in cohort}
    require((len(ids) == 300), ('audit assertion failed'))
    require((sum(x["gold_answerable"] for x in cohort) == 222), ('audit assertion failed'))
    require((all(x["source_type"] == SOURCE_TYPE for x in cohort)), ('audit assertion failed'))

    def one_per_id(rows, field="id"):
        values = [r[field] for r in rows]
        require((len(values) == len(ids)), ('audit assertion failed'))
        require((set(values) == ids), ('audit assertion failed'))
        require((len(values) == len(set(values))), ('audit assertion failed'))

    one_per_id(retrieval)
    one_per_id(router)
    require((Counter(r["variant"] for r in workflow) == {"basic_rag": 300, "controlled_full": 300}), ('audit assertion failed'))
    require((all(sum(r["id"] == uid for r in workflow) == 2 for uid in ids)), ('audit assertion failed'))
    require((Counter(r["model_variant"] for r in generation) == {"base": 300, "sft": 300, "dpo": 300}), ('audit assertion failed'))
    require((all(sum(r["id"] == uid for r in generation) == 3 for uid in ids)), ('audit assertion failed'))
    expected_ablation = {
        "controlled_full", "without_hybrid", "without_reranker",
        "without_router", "without_round2", "sft_instead_of_dpo",
    }
    require((set(r["variant"] for r in ablation) == expected_ablation), ('audit assertion failed'))
    require((len(ablation) == 300 * len(expected_ablation)), ('audit assertion failed'))
    require((all(sum(r["id"] == uid for r in ablation) == len(expected_ablation) for uid in ids)), ('audit assertion failed'))
    require((len(safety) == 300), ('audit assertion failed'))
    require((len(benchmark) == 1250), ('audit assertion failed'))
    require((all(r["source_type"] == SOURCE_TYPE for rows in
               (retrieval, router, workflow, generation, ablation, safety, benchmark)
               for r in rows)), ('audit assertion failed'))

    # No public raw text is accidentally placed into the anonymized evidence pack.
    require((all(q["query_text_published"] is False and q["manual_text_published"] is False
               for q in cohort)), ('audit assertion failed'))
    require((all(r["answer_text_published"] is False and r["evidence_text_published"] is False
               for r in generation)), ('audit assertion failed'))

    recomputed = {
        "retrieval": summarize_retrieval(retrieval),
        "router": summarize_router(router),
        "round2": summarize_round2(workflow),
        "generation": summarize_generation(generation),
        "end_to_end": summarize_end_to_end(workflow, seed=113, resamples=4000),
        "ablation": summarize_outcome_ablation(ablation, seed=125, resamples=4000),
        "safety": summarize_safety(safety),
        "benchmark": summarize_benchmark(benchmark),
    }
    for key, value in recomputed.items():
        assert_equal(summary[key], value, key)

    # The full ablation baseline must be identical to controlled_full E2E success per ID.
    full = {r["id"]: r for r in workflow if r["variant"] == "controlled_full"}
    full_ablation = {r["id"]: r for r in ablation if r["variant"] == "controlled_full"}
    for uid in ids:
        row = full[uid]
        expected = (
            row["final_status"] == "abstained"
            if not row["gold_answerable"]
            else row["final_status"] == "answered"
                 and row["answer_correct"] is True
                 and row["claim_supported"] is True
                 and row["citation_valid"] is True
        )
        require((full_ablation[uid]["grounded_success"] is expected), ('audit assertion failed'))

    # There must be no control invariant violation in the full pipeline.
    require((summary["round2"]["final_gate_violation_count"] == 0), ('audit assertion failed'))

    # Report discloses its evidence boundary and unverified hardware details.
    result_md = (root / "RESULTS.md").read_text(encoding="utf-8")
    require("real_audited" in result_md, "source label missing from report")
    require("维护者声明" in result_md and "独立" in result_md, "provenance caveat missing")
    require("physical_hardware_measured=false" in result_md, "GPU caveat missing")

    # README snapshot is mechanically tied to the same summary.
    project_root = root.parents[3]
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    b = summary["end_to_end"]["variants"]["basic_rag"]
    f = summary["end_to_end"]["variants"]["controlled_full"]
    r = summary["retrieval"]
    rt = summary["router"]
    g = summary["generation"]
    required_fragments = [
        RUN_ID,
        f"{b['grounded_success_per_input']*100:.1f}%",
        f"{f['grounded_success_per_input']*100:.1f}%",
        f"{r['child_recall_at_10']['bm25']*100:.1f}%",
        f"{r['child_recall_at_10']['hybrid']*100:.1f}%",
        f"{rt['learned']['macro_f1']:.3f}",
        f"{g['dpo']['faithfulness_given_answered']*100:.1f}%",
        f"{f['unsafe_answer_per_unanswerable']*100:.1f}%",
        f"{f['p95_latency_ms']/1000:.2f}s",
    ]
    for fragment in required_fragments:
        require((fragment in readme), (f"README not synchronized: {fragment}"))

    return {
        "audit_passed": True,
        "scope": "maintainer-supplied records: hashes and statistical consistency only",
        "source_type": SOURCE_TYPE,
        "provenance_independently_verified": False,
        "real_system_verified": False,
        "run_id": RUN_ID,
        "query_count": len(ids),
        "hashed_files": len(manifest["files"]),
        "workflow_records": len(workflow),
        "generation_records": len(generation),
        "ablation_records": len(ablation),
        "safety_records": len(safety),
        "benchmark_records": len(benchmark),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "docs/experiments/published_runs/RUN-EXAMPLE-2026-001"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
