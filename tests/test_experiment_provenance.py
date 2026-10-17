"""Published record catalog is tied to maintainers claim and hashed files."""
from pathlib import Path
import json
import math
from eval.results.audit_published_run import audit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "docs/experiments/published_runs/RUN-EXAMPLE-2026-001"
CATALOG = ROOT / "docs/experiments/results_catalog.json"


def test_result_catalog_origin_has_owner_attribution():
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert cat["run_id"] == "RUN-EXAMPLE-2026-001"
    assert cat["status"] == "real_audited"
    assert cat["source_type_claimant"] == "project_maintainer"
    assert cat["provenance_independently_verified"] is False
    assert "not demonstrated" in cat["disclosure"]
    assert cat["human_annotation_collected"] is False
    assert cat["model_execution_observed"] is False
    assert cat["gpu_execution_observed"] is False
    assert len(cat["metrics"]) == 4
    for row in cat["metrics"]:
        assert (ROOT / "docs/experiments" / row["source_path"]).is_file()
    assert (ROOT / "docs/experiments" / cat["evidence_manifest"]).is_file()


def test_catalog_values_match_current_published_run():
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    assert cat["run_id"] == summary["run_id"]
    assert audit(RUN)["audit_passed"]
    e = summary["end_to_end"]["variants"]
    r = summary["retrieval"]
    rt = summary["router"]
    expected = {
        "grounded_success_per_input": (
            e["basic_rag"]["grounded_success_per_input"],
            e["controlled_full"]["grounded_success_per_input"]),
        "parent_recall_at_10": (
            r["parent_recall_at_10"]["bm25"],
            r["parent_recall_at_10"]["hybrid"]),
        "router_macro_f1": (rt["rule"]["macro_f1"], rt["learned"]["macro_f1"]),
        "reranker_ndcg_at_5": (r["reranker_ndcg_at_5"]["before"],
                                  r["reranker_ndcg_at_5"]["after"]),
    }
    assert {m["name"] for m in cat["metrics"]} == set(expected)
    for row in cat["metrics"]:
        before, after = expected[row["name"]]
        assert row["paired_n"] == 300
        assert math.isclose(row["baseline"], before, abs_tol=1e-12)
        assert math.isclose(row["candidate"], after, abs_tol=1e-12)


def test_public_docs_link_current_bundle_not_legacy_run():
    docs = [ROOT / "README.md", ROOT / "docs/experiments/README.md"]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert "RUN-EXAMPLE-2026-001" in text
        assert "维护者声明" in text
        assert "example_run/" not in text
