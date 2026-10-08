# -*- coding: utf-8 -*-
"""
Visual Retriever: 多模态图片检索器

使用 Chinese-CLIP (OFA-Sys/chinese-clip-vit-base-patch16) 将图片和文本映射到
同一 512 维向量空间，支持用自然语言查询检索语义匹配的图片。

模型运行在 CPU 上，不占用 GPU 显存。
"""
import os, time, logging, hashlib
from typing import Optional

import numpy as np
from PIL import Image
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema,
    DataType, Collection,
)
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from src import constant
from src.client.mongodb_config import MongoConfig

logger = logging.getLogger(__name__)

# ---- 常量 ----
VISUAL_COL_NAME = "visual_clip"
MAX_PATH_LENGTH = 512
ID_MAX_LENGTH = 100
_visual_dim = 512  # Chinese-CLIP 输出 512 维

# ---- 懒加载全局状态 ----
_model = None
_tokenizer = None
_processor = None
_milvus_connected = False
_mongo_coll = None


def _lazy_init():
    global _milvus_connected, _mongo_coll
    if not _milvus_connected:
        connections.connect(uri=constant.milvus_db_path)
        _milvus_connected = True
    if _mongo_coll is None:
        _mongo_coll = MongoConfig.get_collection("manual_images")


def _load_model(model_path: Optional[str] = None):
    global _model, _tokenizer, _processor
    if _model is None:
        if model_path is None:
            model_path = getattr(constant, 'jina_clip_model_path',
                                 os.path.join(constant.base_dir,
                                              "models/OFA-Sys/chinese-clip-vit-base-patch16"))
        logger.info(f"Loading Chinese-CLIP from: {model_path}")
        _model = AutoModel.from_pretrained(model_path, local_files_only=True)
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        logger.info("Chinese-CLIP loaded (CPU)")


def _encode_text(text: str) -> np.ndarray:
    inputs = _tokenizer([text], padding=True, truncation=True,
                        return_tensors="pt", max_length=52)
    import torch
    with torch.no_grad():
        features = _model.get_text_features(**inputs)
    vec = features.numpy()
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def _encode_image(image_path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = _processor(images=img, return_tensors="pt")
        import torch
        with torch.no_grad():
            features = _model.get_image_features(**inputs)
        vec = features.numpy()
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)
    except Exception as e:
        logger.warning(f"Failed to encode {image_path}: {e}")
        return None


class VisualRetriever:
    """图片向量检索器"""

    def __init__(self, image_paths: Optional[list[str]] = None,
                 retrieve: bool = False,
                 model_path: Optional[str] = None):
        _lazy_init()
        _load_model(model_path)

        fields = [
            FieldSchema(name="image_id", dtype=DataType.VARCHAR,
                        is_primary=True, max_length=ID_MAX_LENGTH),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR,
                        max_length=MAX_PATH_LENGTH),
            FieldSchema(name="visual_vector", dtype=DataType.FLOAT_VECTOR,
                        dim=_visual_dim),
        ]
        schema = CollectionSchema(fields)

        if not retrieve and utility.has_collection(VISUAL_COL_NAME):
            Collection(VISUAL_COL_NAME).drop()

        self.col = Collection(VISUAL_COL_NAME, schema, consistency_level="Strong")
        self.col.create_index("visual_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
        self.col.load()

        if not retrieve and image_paths:
            self._build_index(image_paths)

    def _build_index(self, image_paths: list[str]):
        batch_size = 20
        logger.info(f"Building visual index: {len(image_paths)} images")
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            ids, paths, vectors = [], [], []
            for p in batch:
                vec = _encode_image(p)
                if vec is not None:
                    ids.append(hashlib.md5(p.encode()).hexdigest()[:ID_MAX_LENGTH])
                    paths.append(p)
                    vectors.append(vec[0].tolist())
            if vectors:
                self.col.insert([ids, paths, vectors])
        self.col.flush()
        logger.info(f"Visual index: {self.col.num_entities} entities")

    def retrieve_topk(self, query: str, topk: int = 5) -> list[Document]:
        t0 = time.time()
        qvec = _encode_text(query)[0].tolist()
        results = self.col.search(
            [qvec], anns_field="visual_vector", limit=topk,
            output_fields=["image_id", "image_path"],
            param={"metric_type": "IP", "params": {}},
        )
        docs = []
        for hit in results[0]:
            img_path = hit.entity.get("image_path", "")
            score = hit.distance
            mongo_info = _mongo_coll.find_one({"image_path": img_path}) or {}
            metadata = {
                "image_path": img_path,
                "visual_score": round(float(score), 4),
                "page": mongo_info.get("page"),
                "title": mongo_info.get("title", ""),
                "caption": mongo_info.get("caption", ""),
                "surrounding_text": mongo_info.get("surrounding_text", ""),
            }
            parts = [v for v in [metadata["caption"], metadata["title"],
                                 metadata["surrounding_text"]] if v]
            doc = Document(page_content="\n".join(parts), metadata=metadata)
            docs.append(doc)
        logger.info(f"Visual search '{query[:30]}' → {len(docs)} results ({time.time()-t0:.2f}s)")
        return docs
