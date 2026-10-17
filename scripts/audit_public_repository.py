"""Conservative inspection of source trees or clean GitHub exports.

This complements (but cannot replace) gitignore, history inspection, a true
secret scanner, and review of tracked/staged paths. It makes no authenticity
claims about the experiment included in the repository.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {
    "data", "datasets", "models", "weights", "checkpoints", "processed_docs",
    "saved_index", "artifacts", "outputs", "verified_runs", "benchmark_data",
    "traces", "telemetry", "wandb", "mlruns", "tensorboard",
}
PRIVATE_EXTENSIONS = {
    ".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".faiss",
    ".pkl", ".pickle", ".db", ".sqlite", ".sqlite3", ".pdf", ".docx",
    ".xlsx", ".pptx", ".parquet", ".p12", ".pfx", ".jks", ".kdbx",
    ".pem", ".key", ".bin", ".zip", ".tar", ".gz", ".7z", ".rar", ".csv",
}
PRIVATE_FILENAMES = {
    "credentials.json", "secrets.json", "id_rsa", "id_ed25519", ".npmrc",
    ".pypirc", "service-account.json", "service_account.json", "kubeconfig",
}
PUBLIC_ENV_NAMES = {".env.example", ".env.dpo.example"}
IGNORED_SCAN_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
REQUIRED_IGNORES = {
    "/data/", "/models/", "/outputs/", "/datasets/", "/eval/datasets/",
    "/eval/reports/", "/docs/experiments/reference_results/*",
}
PUBLISHED_JSONL = {
    "cohort_anonymized.jsonl", "retrieval_anonymized.jsonl", "router_anonymized.jsonl",
    "workflow_anonymized.jsonl", "generation_judgements.jsonl",
    "ablation_outcomes.jsonl", "safety_anonymized.jsonl", "benchmark_anonymized.jsonl",
}
# High-confidence token formats, rather than pattern matching every variable named 'api_key'.
SENSITIVE_CONTENT = [
    ("PEM private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
]
ENV_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:OPENAI_API_KEY|GITHUB_TOKEN|AWS_SECRET_ACCESS_KEY|"
    r"(?:[A-Z0-9_]*_)?(?:API_KEY|CLIENT_SECRET|PASSWORD))\s*[=:]\s*[\"']?"
    r"([^\s\"'#,]+)")
PLACEHOLDERS = {"EMPTY", "EXAMPLE", "REPLACE_ME", "YOUR_KEY", "CHANGEME", "NONE", "NULL", "DUMMY", "TEST"}


def _sensitive_content(path: Path) -> list[str]:
    if path.stat().st_size > 2_000_000:
        return []  # public file-size policy rejects before this
    raw = path.read_bytes()
    if raw.startswith(b"%PDF-"):
        return ["PDF binary content"]
    if b"\x00" in raw:
        return []  # don't parse arbitrary binaries as code; extension policy applies
    text = raw.decode("utf-8", errors="replace")
    findings = [name for name, pattern in SENSITIVE_CONTENT if pattern.search(text)]
    if path.name.startswith(".env") or path.suffix in {".ini", ".cfg", ".yml", ".yaml", ".toml"}:
        for m in ENV_SECRET_ASSIGNMENT.finditer(text):
            value = m.group(1).strip().strip("\"'")
            if (value.upper() not in PLACEHOLDERS and not value.startswith("${") and
                    not value.startswith("<") and len(value) > 7 and
                    not value.startswith(("/", "http://127.0.0.1", "http://localhost"))):
                findings.append("suspicious environment secret assignment")
                break
    return findings


def audit_public_repository(root: Path = ROOT, *, exported: bool = False) -> dict:
    root = Path(root)
    errors: list[str] = []
    files_examined = 0
    if not root.is_dir():
        return {"passed": False, "files_examined": 0, "errors": [f"project missing: {root}"]}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        parts = relative.parts
        if any(part in IGNORED_SCAN_DIRS for part in parts):
            continue
        # During development these directories may exist. In --export mode, all
        # artifacts in the proposed public checkout must be examined and rejected.
        if not exported and (parts[0] in FORBIDDEN_ROOTS or
                             parts[:2] == ("eval", "reports")):
            continue
        if path.is_symlink():
            errors.append(f"symlink not permitted in public release: {relative}")
            continue
        if path.is_dir():
            continue
        files_examined += 1
        if parts[0] in FORBIDDEN_ROOTS or parts[:2] == ("eval", "reports"):
            errors.append(f"forbidden data directory: {relative}")
        if relative.suffix.lower() in PRIVATE_EXTENSIONS or relative.name.endswith(".tfevents"):
            errors.append(f"private binary/document file: {relative}")
        if relative.name.startswith(".env") and relative.name not in PUBLIC_ENV_NAMES:
            errors.append(f"private environment file: {relative}")
        if relative.name in PRIVATE_FILENAMES:
            errors.append(f"credential file: {relative}")
        if parts[:3] == ("docs", "experiments", "reference_results"):
            errors.append(f"untracked raw reference archive embedded: {relative}")
        if parts[:2] == ("eval", "data") and len(parts) == 3:
            allowed = (relative.name.endswith("_synthetic.jsonl") or
                       relative.name.endswith("_synthetic.json") or
                       relative.name in {"SCHEMAS.md", "ANNOTATION_PROTOCOL.md"})
            if not allowed:
                errors.append(f"non-fixture benchmark file in public eval/data: {relative}")
            if path.stat().st_size > 50_000:
                errors.append(f"fixture exceeds 50KB limit: {relative}")
        is_published_anonymized = (
            parts[:3] == ("docs", "experiments", "published_runs") and
            len(parts) == 5 and relative.name in PUBLISHED_JSONL and
            (path.parent / "manifest.json").is_file()
        )
        if relative.suffix == ".jsonl" and not (
            parts[:2] == ("eval", "data") or parts[0] == "examples" or is_published_anonymized
        ):
            errors.append(f"raw JSONL outside documented fixtures/published bundle: {relative}")
        if path.stat().st_size > 1_000_000:
            errors.append(f"file exceeds public-package threshold (1MB): {relative}")
        for reason in _sensitive_content(path):
            errors.append(f"{reason}: {relative}")
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        missing = REQUIRED_IGNORES - set(gitignore.read_text(encoding="utf-8").splitlines())
        if missing:
            errors.append("missing .gitignore rules: " + ", ".join(sorted(missing)))
    else:
        errors.append("missing .gitignore")
    return {"passed": not errors, "files_examined": files_examined, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit files intended for the public repository")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--export", action="store_true", help="Validate a cleaned public checkout/ZIP tree")
    args = parser.parse_args()
    result = audit_public_repository(args.root, exported=args.export)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
