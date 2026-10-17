# Security policy

This repository is a reference implementation with a **synthetic/demo-only** FastAPI entry point.
It is not a secured public production API. Do not expose `/chat` without deployment-specific
identity, authorization, transport deadlines, request quotas and URI redaction.

Please report a potential vulnerability privately to the repository maintainer using
GitHub's private vulnerability reporting when enabled. **Do not publish keys, tokens,
private PDFs, model weights, or customer traces in issues.**

Supported runtime: Python 3.10 and 3.12 in CI. Security fixes are delivered on the main branch.
