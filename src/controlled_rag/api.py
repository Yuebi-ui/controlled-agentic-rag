"""Opt-in synthetic FastAPI example. REAL mode must be explicitly wired later."""


import os
import hmac
from typing import Callable

from pydantic import BaseModel, Field

from .demo import build_demo_pipeline, demo_request
from .admission import AdmissionController, BusyError


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


def _demo_public_payload(payload: dict) -> dict:
    """Only intentionally fictional demo:// URIs are safe in the demo response."""
    for citation in payload.get("citations", []):
        for field in ("source_uri", "image_uri"):
            value = citation.get(field)
            if value and not value.startswith("demo://"):
                citation[field] = None
    return payload


def create_app(factory: Callable = build_demo_pipeline, *, demo_api_key: str | None = None):
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Controlled Agentic RAG — synthetic demo", version="0.1.0")
    pipeline = factory()
    configured_key = demo_api_key if demo_api_key is not None else os.getenv("RAG_DEMO_API_KEY")
    if configured_key is not None and (not configured_key.strip() or configured_key == "EMPTY"):
        raise ValueError("RAG_DEMO_API_KEY must not be blank or a placeholder")
    admission = AdmissionController()  # demo process only; tune and benchmark for production

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "synthetic-demo", "production_ready": False}

    @app.post("/chat")
    async def chat(request: ChatRequest, http_request: Request):
        # No authentication by default ONLY for localhost/testclient. For any
        # non-local caller an explicit demo token is mandatory. This is not a
        # multi-tenant production authentication scheme.
        remote = http_request.client.host if http_request.client else None
        if configured_key is None:
            if remote not in {"127.0.0.1", "::1", "testclient"}:
                raise HTTPException(status_code=403, detail="demo is localhost-only without an API key")
        else:
            expected = f"Bearer {configured_key}"
            provided = http_request.headers.get("authorization", "")
            if not hmac.compare_digest(provided, expected):
                raise HTTPException(status_code=401, detail="invalid demo token")
        # The HTTP client must not provide its own authorization / version context.
        try:
            async with admission.slot():
                result = await run_in_threadpool(pipeline.run, request.query, demo_request())
        except BusyError:
            raise HTTPException(status_code=429, detail="demo server is busy")
        if result.final_status == "failed":
            raise HTTPException(status_code=503, detail={"code": "demo_pipeline_failed"})
        return _demo_public_payload(result.model_dump())

    return app


def _from_environment():
    if os.getenv("RAG_MODE") != "demo":
        raise RuntimeError("No real serving stack is configured. Set RAG_MODE=demo for a SYNTHETIC sample API.")
    return create_app()


# Lazy FastAPI CLI import via `controlled_rag.api:app` is intentionally demo-only.
# If RAG_MODE is not set, import fails closed instead of claiming real serving works.
if os.getenv("RAG_MODE") == "demo":
    app = create_app()
