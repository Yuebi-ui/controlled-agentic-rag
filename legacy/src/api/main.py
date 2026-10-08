# -*- coding: utf-8 -*-
# FastAPI 服务层 — Agentic RAG API
#
# 启动: uvicorn src.api.main:app --host 0.0.0.0 --port 9000

import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from src.pipeline.rag_pipeline import RAGPipeline
from src.api.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Agentic RAG API", version="1.0")

# 全局管线实例（进程启动时 warmstart）
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """非流式问答"""
    pipeline = get_pipeline()
    result = pipeline.run(req.query, stream=False)

    return ChatResponse(
        answer=result["answer"],
        cite_pages=result.get("cite_pages", []),
        related_images=result.get("related_images", []),
        citations=result.get("citation_details", []),
        citation_verified=result.get("citation_verified", True),
        citation_unsupported=result.get("citation_unsupported", 0),
        rewritten_query=result.get("rewritten_query", ""),
        query_type=result.get("query_type", ""),
        intent=result.get("intent", "knowledge_qa"),
        evidence_enough=result.get("evidence_enough", True),
        evidence_reason=result.get("evidence_reason", ""),
        self_rag_triggered=result.get("self_rag", False),
        elapsed=result.get("total_time", 0.0),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答"""
    pipeline = get_pipeline()
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_token(token: str):
        loop.call_soon_threadsafe(queue.put_nowait, token)

    async def event_generator():
        # 后台运行管线
        loop = asyncio.get_running_loop()
        result_holder = {}

        def run_pipeline():
            result_holder["result"] = pipeline.run(
                req.query, stream=True, stream_callback=on_token,
            )

        task = loop.run_in_executor(None, run_pipeline)

        # 流式输出 token
        while not task.done() or not queue.empty():
            try:
                token = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                continue

        await task

        # 最终结果
        result = result_holder.get("result", {})
        yield f"data: {json.dumps({'done': True, 'answer': result.get('answer', ''), 'cite_pages': result.get('cite_pages', []), 'related_images': result.get('related_images', []), 'elapsed': result.get('total_time', 0)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
