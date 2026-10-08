# -*- coding: utf-8 -*-
# infer_agentic.py — Agentic Hybrid RAG 交互入口
#
# 管线：
#   QueryAgent → Dynamic TopK → RetrievalAgent → Reranker
#   → EvidenceAgent → (Self-RAG 二次检索) → AnswerAgent → CitationVerifier → post_processing
#
# 使用共享管线 src/pipeline/rag_pipeline.py

import time
from src.pipeline.rag_pipeline import RAGPipeline

SEPARATOR = "=" * 80


def main():
    print("正在初始化组件...")
    pipeline = RAGPipeline()
    print("初始化完成。输入问题开始，输入 quit 退出。\n")

    while True:
        try:
            query = input("输入 → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出。")
            break

        process_query(query, pipeline)


def process_query(query: str, pipeline: RAGPipeline):
    t0 = time.time()

    print(f"\n{SEPARATOR}")
    print("【Agentic RAG 管线】")

    # 流式回调：逐字打印
    print("答案 → ", end="", flush=True)
    answer_holder = {"response": ""}

    def print_token(token: str):
        print(token, end="", flush=True)
        answer_holder["response"] += token

    result = pipeline.run(query, stream=True, stream_callback=print_token)

    response = result.get("raw_answer", answer_holder["response"])
    print(f"\n  生成耗时: {time.time() - t0:.2f}s")

    # Citation Verification
    print(f"\n【Citation Verifier】引用校验")
    cv_details = result.get("citation_details", [])
    if not cv_details:
        print("  无引用标记，跳过校验")
    else:
        status_map = {"support": "支持", "partial": "部分支持", "not_support": "不支持",
                      "unknown": "未知", "no_citation": "无引用", "invalid_citation": "无效引用"}
        for i, cr in enumerate(cv_details):
            status = status_map.get(cr.get("support_status", ""), cr.get("support_status", ""))
            ids = ",".join(str(x) for x in cr.get("cited_doc_ids", [])) or "无"
            claim = cr.get("claim", "")[:60]
            print(f"  [{i+1}] {status} | 引用:[{ids}] | {claim}...")
        if result.get("citation_verified"):
            print("  结果: 全部通过")
        else:
            print(f"  结果: {result.get('citation_unsupported', 0)} 条未通过")

    # Post Processing
    cite_pages = result.get("cite_pages", [])
    images = result.get("related_images", [])
    print(f"\n【后处理】引用提取")
    print(f"  答案: {result['answer']}")
    print(f"  引用页码: {cite_pages}")
    if images:
        print(f"  相关图片: {len(images)} 张")

    # Summary
    print(f"\n{SEPARATOR}")
    print("【管线摘要】")
    steps = [
        f"Query Rewrite: {query} → {result.get('rewritten_query', '')}",
        f"Query Type: {result.get('query_type', '')}",
        f"Intent: {result.get('intent', '')}",
        f"BM25召回: {result.get('bm25_count', 0)} → "
        f"Milvus召回: {result.get('milvus_count', 0)} → "
        f"合并: {result.get('merged_count', 0)}",
        f"精排: {result.get('ranked_doc_count', 0)} 条",
        f"证据判断: {'足够' if result.get('evidence_enough') else '不足'}",
        f"Self-RAG: {'已触发二次检索' if result.get('self_rag') else '未触发'}",
        f"答案长度: {len(response)} 字符",
        f"总耗时: {result.get('total_time', 0):.2f}s",
    ]
    for s in steps:
        print(f"  - {s}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
