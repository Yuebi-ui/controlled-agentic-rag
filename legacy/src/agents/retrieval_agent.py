# -*- coding: utf-8 -*-
# RetrievalAgent：BM25 + Milvus 文本 + Milvus 图片 三路混合召回

import time
from concurrent.futures import ThreadPoolExecutor
from src.utils import merge_docs


class RetrievalAgent:
    def __init__(self, bm25_retriever, milvus_retriever, visual_retriever=None):
        self.bm25 = bm25_retriever
        self.milvus = milvus_retriever
        self.visual = visual_retriever  # 可选：多模态图片检索

    def retrieve(self, query: str, bm25_topk: int = 5,
                 milvus_topk: int = 8, visual_topk: int = 3) -> dict:
        t1 = time.time()

        # 并行三路检索
        max_workers = 3 if self.visual else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_bm25 = executor.submit(self.bm25.retrieve_topk, query, bm25_topk)
            future_milvus = executor.submit(self.milvus.retrieve_topk, query, milvus_topk)
            future_visual = (
                executor.submit(self.visual.retrieve_topk, query, visual_topk)
                if self.visual else None
            )

            bm25_docs = future_bm25.result()
            milvus_docs = future_milvus.result()
            visual_docs = future_visual.result() if future_visual else []

        t2 = time.time()
        merged_docs = merge_docs(bm25_docs, milvus_docs)

        return {
            "bm25_docs": bm25_docs,
            "milvus_docs": milvus_docs,
            "visual_docs": visual_docs,      # 图片检索结果（不参与文本 merge）
            "merged_docs": merged_docs,       # 文本检索 merge 结果
            "elapsed": t2 - t1,
        }
