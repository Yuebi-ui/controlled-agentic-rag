# -*- coding: utf-8 -*-
# API Pydantic schemas

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    stream: bool = Field(default=False, description="是否流式返回")


class CitationItem(BaseModel):
    claim: str
    cited_doc_ids: list[int]
    support_status: str
    reason: str


class ImageItem(BaseModel):
    page: int | None = None
    title: str = ""
    image_path: str = ""
    caption: str = ""
    base64: str = ""
    visual_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    cite_pages: list[int] = Field(default_factory=list)
    related_images: list[ImageItem] = Field(default_factory=list)
    citations: list[CitationItem] = Field(default_factory=list)
    citation_verified: bool = True
    citation_unsupported: int = 0
    rewritten_query: str = ""
    query_type: str = ""
    intent: str = "knowledge_qa"
    evidence_enough: bool = True
    evidence_reason: str = ""
    self_rag_triggered: bool = False
    elapsed: float = 0.0
