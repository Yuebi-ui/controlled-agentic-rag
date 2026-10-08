# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

import re
import base64
import os
import json
from langchain_core.documents import Document
from src.client.mongodb_config import MongoConfig
from src import constant

_manual_collection = None


def _get_manual_collection():
    global _manual_collection
    if _manual_collection is None:
        _manual_collection = MongoConfig.get_collection("manual_text")
    return _manual_collection


def _load_caption_cache() -> dict:
    """加载图片 Caption 缓存"""
    cache_path = getattr(constant, 'image_caption_cache_path',
                         os.path.join(constant.base_dir, "data/image_captions.json"))
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _enrich_image_info(image: dict, caption_cache: dict) -> dict:
    """用 Caption 缓存补充图片元信息（返回副本，不修改原 dict）"""
    if not isinstance(image, dict):
        return image
    image = dict(image)  # 浅拷贝，避免副作用
    if not image.get("caption"):
        img_path = image.get("image_path", "")
        filename = os.path.basename(img_path)
        if filename in caption_cache:
            image["caption"] = caption_cache[filename]
    return image


def merge_docs(docs1, docs2):
    merged_docs = []
    merged_ids = set()
    candidate_docs = docs1 + docs2

    # 收集所有需要查询的parent_id，批量查询MongoDB
    parent_ids = []
    for doc in candidate_docs:
        pid = doc.metadata.get("parent_id")
        if pid:
            parent_ids.append(pid)

    parent_docs_map = {}
    if parent_ids:
        mongo_results = list(_get_manual_collection().find({"unique_id": {"$in": parent_ids}}))
        parent_docs_map = {r["unique_id"]: r for r in mongo_results}

    for doc in candidate_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_mg = parent_docs_map.get(parent_id)
            if parent_mg is None:
                continue
            unique_id = parent_mg["unique_id"]
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                parent_doc = Document(page_content=parent_mg["page_content"], metadata=parent_mg["metadata"])
                merged_docs.append(parent_doc)
        else:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                merged_docs.append(doc)
    return merged_docs


def _encode_image_base64(image_path: str) -> str:
    """将本地图片编码为 base64 data URL"""
    try:
        if not os.path.exists(image_path):
            return ""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif"}
        mime = mime_map.get(ext, "jpeg")
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return ""


def post_processing(response, docs, visual_docs=None):
    """
    答案后处理：提取引用页码、相关图片

    Args:
        response: AnswerAgent 生成的回答文本
        docs: 精排后的文本 Document 列表（用于提取引用页码和图片）
        visual_docs: 图片检索结果 Document 列表（多模态检索结果）

    Returns:
        dict with answer, cite_pages, related_images
    """
    all_cites = re.findall("[【](.*?)[】]", response)
    cites = []
    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    cites = list(set(cites))
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)
    # 清理 Qwen3 thinking 标签（空标签和有内容的都清）
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    related_images = []
    seen_paths = set()
    pages = []
    caption_cache = _load_caption_cache()

    # 1. 从引用文档中提取图片
    for index in cites:
        if index < 1 or index > len(docs):
            continue
        images = docs[index - 1].metadata.get("images_info", [])
        page = docs[index - 1].metadata.get("page")
        if page is not None:
            pages.append(page)
        for image in images:
            image = _enrich_image_info(image, caption_cache)
            if isinstance(image, dict) and (image.get("title") or image.get("caption")):
                img_path = image.get("image_path", "")
                if img_path and img_path not in seen_paths:
                    seen_paths.add(img_path)
                    related_images.append({
                        "page": image.get("page"),
                        "title": image.get("title", ""),
                        "image_path": img_path,
                        "caption": image.get("caption", ""),
                        "base64": _encode_image_base64(img_path),
                    })

    # 2. 从视觉检索结果中补充图片（未被文本引用直接命中的）
    if visual_docs:
        for vdoc in visual_docs:
            img_path = vdoc.metadata.get("image_path", "")
            if img_path and img_path not in seen_paths:
                seen_paths.add(img_path)
                page = vdoc.metadata.get("page")
                if page and page not in pages:
                    pages.append(page)
                # 从缓存补充 caption
                filename = os.path.basename(img_path)
                caption = vdoc.metadata.get("caption", "") or caption_cache.get(filename, "")
                related_images.append({
                    "page": vdoc.metadata.get("page"),
                    "title": vdoc.metadata.get("title", ""),
                    "image_path": img_path,
                    "caption": caption,
                    "visual_score": vdoc.metadata.get("visual_score"),
                    "base64": _encode_image_base64(img_path),
                })

    pages = sorted(list(set(pages)))
    return {
        "answer": answer,
        "cite_pages": pages,
        "related_images": related_images
    }
