"""Public release: private file boundaries and obvious example disclosures."""
import json
from pathlib import Path

from scripts.audit_public_repository import audit_public_repository

ROOT = Path(__file__).resolve().parents[1]


def test_public_repo_contains_only_safe_example_data():
    report = audit_public_repository(ROOT, exported=False)
    assert report["passed"], report["errors"]
    assert not (ROOT / "docs/experiments/reference_results/raw").exists()
    assert (ROOT / "docs/experiments/published_runs/RUN-EXAMPLE-2026-001/manifest.json").exists()
    assert (ROOT / "docs/experiments/published_runs/RUN-EXAMPLE-2026-001/cohort_anonymized.jsonl").exists()


def test_accidental_private_benchmark_is_rejected(tmp_path):
    (tmp_path / ".gitignore").write_text("\n".join([
        "/data/", "/models/", "/outputs/", "/datasets/", "/eval/datasets/",
        "/eval/reports/", "/docs/experiments/reference_results/*",
    ]), encoding="utf-8")
    (tmp_path / "docs/experiments/reference_results/raw").mkdir(parents=True)
    (tmp_path / "docs/experiments/reference_results/raw/workflow.jsonl").write_text("{}\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/private.pdf").write_bytes(b"private")
    errors = audit_public_repository(tmp_path, exported=True)["errors"]
    assert any("reference archive" in err for err in errors)
    assert any("forbidden data directory" in err for err in errors)


def test_minimal_ci_fixtures_remain_allowed(tmp_path):
    (tmp_path / ".gitignore").write_text("\n".join([
        "/data/", "/models/", "/outputs/", "/datasets/", "/eval/datasets/",
        "/eval/reports/", "/docs/experiments/reference_results/*",
    ]), encoding="utf-8")
    folder = tmp_path / "eval/data"
    folder.mkdir(parents=True)
    (folder / "router_synthetic.jsonl").write_text('{"id":"SYN-1"}\n')
    assert audit_public_repository(tmp_path, exported=True)["passed"]


def test_published_run_documentation_and_provenance_caveat():
    assert (ROOT / "docs/experiments/published_runs/RUN-EXAMPLE-2026-001/RESULTS.md").is_file()
    assert "维护者声明" in (ROOT / "README.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "未包含可独立验证的模型执行" in readme
    manifest = json.loads(
        (ROOT / "docs/experiments/published_runs/RUN-EXAMPLE-2026-001/manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["source_type_claimant"] == "project_maintainer"
    assert manifest["provenance_independently_verified"] is False


def test_private_locally_generated_outputs_not_exportable(tmp_path):
    (tmp_path / ".gitignore").write_text("\n".join([
        "/data/", "/models/", "/outputs/", "/datasets/", "/eval/datasets/",
        "/eval/reports/", "/docs/experiments/reference_results/*",
    ]), encoding="utf-8")
    local = tmp_path / "outputs/reference_results/raw"
    local.mkdir(parents=True)
    (local / "gpu.jsonl").write_text("{}\n")
    assert audit_public_repository(tmp_path)["passed"]
    assert not audit_public_repository(tmp_path, exported=True)["passed"]


def test_unexpected_public_pdf_and_real_token_are_rejected(tmp_path):
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text(encoding="utf-8"))
    (tmp_path / "docs").mkdir()
    private = tmp_path / "docs" / "manual.pdf"
    private.write_bytes(b"%PDF-1.5\nsecret manual")
    token = tmp_path / "notes.md"
    token.write_text("token: ghp_" + "A" * 40)
    result = audit_public_repository(tmp_path, exported=True)
    assert not result["passed"]
    assert any("private binary/document" in error for error in result["errors"])
    assert any("GitHub token" in error for error in result["errors"])


def test_renamed_pdf_content_and_symlink_rejected(tmp_path):
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text(encoding="utf-8"))
    (tmp_path / "manual.md").write_bytes(b"%PDF-1.6\nfile")
    (tmp_path / "shortcut").symlink_to("manual.md")
    errors = audit_public_repository(tmp_path, exported=True)["errors"]
    assert any("PDF binary" in error for error in errors)
    assert any("symlink" in error for error in errors)


def test_private_archives_are_rejected(tmp_path):
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text(encoding="utf-8"))
    (tmp_path / "secret.zip").write_bytes(b"PK\x03\x04")
    errors = audit_public_repository(tmp_path, exported=True)["errors"]
    assert any("private binary/document" in error for error in errors)
