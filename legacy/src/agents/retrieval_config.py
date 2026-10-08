# -*- coding: utf-8 -*-
# 动态 TopK 配置：根据问题类型调整召回数量

RETRIEVAL_CONFIG = {
    "fact_qa":   {"bm25_topk": 5,  "milvus_topk": 8,  "rerank_topk": 4},
    "compare":   {"bm25_topk": 10, "milvus_topk": 15, "rerank_topk": 6},
    "summary":   {"bm25_topk": 12, "milvus_topk": 20, "rerank_topk": 8},
    "multi_hop": {"bm25_topk": 15, "milvus_topk": 25, "rerank_topk": 10},
    "other":     {"bm25_topk": 5,  "milvus_topk": 8,  "rerank_topk": 4},
}


def get_retrieval_config(query_type: str, evidence_enough: bool = True) -> dict:
    config = RETRIEVAL_CONFIG.get(query_type, RETRIEVAL_CONFIG["other"]).copy()
    if not evidence_enough:
        config["bm25_topk"] = int(config["bm25_topk"] * 1.5)
        config["milvus_topk"] = int(config["milvus_topk"] * 1.5)
        config["rerank_topk"] = config["rerank_topk"] + 2
    return config
