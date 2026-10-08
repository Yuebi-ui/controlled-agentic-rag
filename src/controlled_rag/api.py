"""Opt-in synthetic FastAPI example. REAL mode must be explicitly wired later."""

from __future__ import annotations

import os
from typing import Callable

from pydantic import BaseModel, Field

from .demo import build_demo_pipeline, demo_request
from .admission import AdmissionController, BusyError


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


def create_app(factory: Callable = build_demo_pipeline):
    from fastapi import FastAPI, HTTPException
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Controlled Agentic RAG — synthetic demo", version="0.1.0")
    pipeline = factory()
    admission = AdmissionController()  # demo process only; tune and benchmark for production

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "synthetic-demo", "production_ready": False}

    @app.post("/chat")
    async def chat(request: ChatRequest):
        # The HTTP client must not provide its own authorization / version context.
        try:
            async with admission.slot():
                result = await run_in_threadpool(pipeline.run, request.query, demo_request())
        except BusyError:
            raise HTTPException(status_code=429, detail="demo server is busy")
        if result.final_status == "failed":
            raise HTTPException(status_code=503, detail=result.model_dump())
        return result.model_dump()

    return app


def _from_environment():
    if os.getenv("RAG_MODE") != "demo":
        raise RuntimeError("No real serving stack is configured. Set RAG_MODE=demo for a SYNTHETIC sample API.")
    return create_app()


# Lazy FastAPI CLI import via `controlled_rag.api:app` is intentionally demo-only.
# If RAG_MODE is not set, import fails closed instead of claiming real serving works.
if os.getenv("RAG_MODE") == "demo":
    app = create_app()
