# -*- coding: utf-8 -*-
# 消融评测配置：定义各管道变体

# 5 个管道变体，逐级叠加组件
ABLATION_VARIANTS = {
    "bm25_only": {
        "label": "BM25 Only",
        "use_bm25": True,
        "use_milvus": False,
        "use_reranker": False,
        "use_query_agent": False,
        "use_evidence_agent": False,
        "use_self_rag": False,
    },
    "milvus_only": {
        "label": "Milvus Only",
        "use_bm25": False,
        "use_milvus": True,
        "use_reranker": False,
        "use_query_agent": False,
        "use_evidence_agent": False,
        "use_self_rag": False,
    },
    "hybrid": {
        "label": "Hybrid (BM25 + Milvus)",
        "use_bm25": True,
        "use_milvus": True,
        "use_reranker": False,
        "use_query_agent": False,
        "use_evidence_agent": False,
        "use_self_rag": False,
    },
    "hybrid_rerank": {
        "label": "Hybrid + Reranker",
        "use_bm25": True,
        "use_milvus": True,
        "use_reranker": True,
        "use_query_agent": False,
        "use_evidence_agent": False,
        "use_self_rag": False,
    },
    "agentic_rag": {
        "label": "Agentic RAG (Full)",
        "use_bm25": True,
        "use_milvus": True,
        "use_reranker": True,
        "use_query_agent": True,
        "use_evidence_agent": True,
        "use_self_rag": True,
    },
}

# 非 agentic 变体使用固定检索参数（与 final_score.py 一致）
DEFAULT_BM25_TOPK = 5
DEFAULT_MILVUS_TOPK = 10
DEFAULT_RERANK_TOPK = 5

# 默认评测参数
DEFAULT_LIMIT = 50
DATA_PATH = "data/qa_pairs/test_qa_pair_verify.json"
OUTPUT_DIR = "data/eval_reports"
CACHE_DIR = "data/eval_reports/cache"
