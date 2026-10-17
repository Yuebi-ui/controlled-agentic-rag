"""Maintainer-declared published bundle is internally auditable and synchronized."""
from pathlib import Path

from eval.results.audit_published_run import audit
from scripts.audit_public_repository import audit_public_repository

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "docs/experiments/published_runs/RUN-EXAMPLE-2026-001"


def test_reference_published_run_audits():
    result = audit(RUN)
    assert result["audit_passed"] is True
    assert result["real_system_verified"] is False
    assert result["query_count"] == 300
    assert result["workflow_records"] == 600
    assert result["generation_records"] == 900
    assert result["ablation_records"] == 1800
    assert result["benchmark_records"] == 1250


def test_reference_published_run_has_complete_public_evidence_files():
    required = {
        "RESULTS.md", "manifest.json", "summary.json",
        "cohort_anonymized.jsonl", "retrieval_anonymized.jsonl",
        "router_anonymized.jsonl", "workflow_anonymized.jsonl",
        "generation_judgements.jsonl", "ablation_outcomes.jsonl",
        "safety_anonymized.jsonl", "benchmark_anonymized.jsonl",
    }
    assert required <= {p.name for p in RUN.iterdir() if p.is_file()}


def test_public_export_policy_accepts_only_documented_anonymized_bundle():
    report = audit_public_repository(ROOT, exported=False)
    assert report["passed"], report["errors"]


def test_readme_headline_links_to_reference_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "RUN-EXAMPLE-2026-001" in readme
    assert "Unanswerable false-answer rate" in readme
    assert "P95 E2E latency" in readme
    assert "维护者声明" in readme


def test_published_run_tamper_is_detected(tmp_path):
    import shutil
    import pytest
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    target = copied / "workflow_anonymized.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA mismatch"):
        audit(copied)


def test_published_run_cannot_be_silently_relabelled_synthetic(tmp_path):
    import json
    import shutil
    import pytest
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    target = copied / "manifest.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["source_type"] = "incorrectly_relabelled"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssertionError):
        audit(copied)
