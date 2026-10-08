# -*- coding: utf-8 -*-
# Web Demo：Gradio 可视化 Agentic RAG 完整管线（含图片展示）

import gradio as gr

from src.pipeline.rag_pipeline import RAGPipeline


def _format_docs(docs, n):
    lines = []
    for i, doc in enumerate(docs[:n]):
        text = doc.page_content[:200].replace("\n", " ")
        lines.append(f"[{i+1}] {text}...")
    return "\n\n".join(lines) if lines else "无"


def _format_ranked(docs, scores):
    lines = []
    for i, (doc, score) in enumerate(zip(docs, scores)):
        text = doc.page_content[:150].replace("\n", " ")
        lines.append(f"[{i+1}] score={score:.4f} | {text}...")
    return "\n\n".join(lines) if lines else "无"


def _format_citations(claim_results):
    if not claim_results:
        return "无引用标记，未校验"
    lines = []
    status_map = {"support": "支持", "partial": "部分支持", "not_support": "不支持",
                  "unknown": "未知", "no_citation": "无引用", "invalid_citation": "无效引用"}
    for i, cr in enumerate(claim_results):
        status = status_map.get(cr["support_status"], cr["support_status"])
        ids = ",".join(str(x) for x in cr.get("cited_doc_ids", [])) or "无"
        lines.append(f"[{i+1}] {status} | 引用: [{ids}] | {cr['claim'][:80]}...")
    return "\n".join(lines)


import os
import traceback

# Gradio UI
pipeline = RAGPipeline()

css = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.answer-box textarea { font-size: 16px !important; line-height: 1.8 !important; }
.stage-title { font-size: 14px; font-weight: bold; color: #555; margin-top: 8px; }
.footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
"""


def ask(query):
    if not query.strip():
        empty = [""] * 23
        empty[2] = []  # gallery 在 outputs 的第 3 位（index 2）
        return empty
    r = pipeline.run(query)

    rank_docs = r.get("_ranked_docs", [])
    rank_scores = r.get("_ranked_scores", [])
    bm25_docs = r.get("_bm25_docs", [])
    milvus_docs = r.get("_milvus_docs", [])
    visual_docs = r.get("_visual_docs", [])
    images = r.get("related_images", [])
    cite_pages = r.get("cite_pages", [])

    evidence_enough = "足够" if r.get("evidence_enough") else "不足"
    self_rag = "已触发" if r.get("self_rag") else "未触发"
    if r.get("citation_verified"):
        citation_verified = "全部通过"
    else:
        citation_verified = f"{r.get('citation_unsupported', 0)} 条未通过"

    # 构建图片 Gallery 数据：[(path, caption), ...]
    # 注意：Gradio Gallery 只接受文件路径，不能使用 base64 data URI
    gallery_data = []
    for img in images:
        path = img.get("image_path", "")
        label = img.get("title") or img.get("caption", "") or os.path.basename(path)
        if path:
            gallery_data.append((path, label))

    visual_info = f"图片检索: {len(visual_docs)} 张  |  图片展示: {len(images)} 张"

    return [
        r.get("answer", ""),
        f"引用页码: {cite_pages}  |  {visual_info}  |  总耗时: {r.get('total_time', 0):.2f}s",
        gallery_data,
        r.get("rewritten_query", query),
        r.get("query_type", ""),
        r.get("intent", ""),
        r.get("rewrite_reason", ""),
        r.get("bm25_count", 0),
        r.get("milvus_count", 0),
        len(visual_docs),
        r.get("merged_count", 0),
        r.get("retrieval_time", "0.00s"),
        _format_ranked(rank_docs, rank_scores),
        evidence_enough,
        r.get("evidence_reason", ""),
        self_rag,
        r.get("suggested_query", ""),
        r.get("second_merged_count", ""),
        citation_verified,
        _format_citations(r.get("citation_details", [])),
        _format_docs(bm25_docs, 3),
        _format_docs(milvus_docs, 3),
        f"{r.get('total_time', 0):.2f}s",
    ]


with gr.Blocks(title="Agentic Hybrid RAG — 特斯拉 Model 3 手册问答", css=css) as demo:
    gr.Markdown("""
    # Agentic Hybrid RAG — 特斯拉 Model 3 用户手册问答
    **管线**: Query Rewrite → Dynamic TopK → BM25 + Milvus 混合检索 → Reranker 精排 → 证据判断 → Self-RAG → 答案生成 → Citation Verify
    """)

    with gr.Row():
        inp = gr.Textbox(
            label="输入问题",
            placeholder="例如：离车后自动上锁怎么设置？",
            scale=5,
        )
        btn = gr.Button("提问", variant="primary", scale=1)

    # ====== 最终答案 + 图片 ======
    gr.Markdown("### 回答")
    answer = gr.Textbox(label="答案内容", lines=6, elem_classes=["answer-box"])
    answer_meta = gr.Textbox(label="引用 / 图片 / 耗时", lines=1)

    gr.Markdown("### 相关图片")
    gallery = gr.Gallery(
        label="手册中的相关图片",
        columns=3,
        height=300,
        object_fit="contain",
    )

    # ====== 管线详情（折叠） ======
    with gr.Accordion("管线中间过程", open=False):
        gr.Markdown("#### Step 1 — Query Rewrite（查询改写与分类）")
        with gr.Row():
            rewritten = gr.Textbox(label="改写后查询", lines=2, scale=1)
            query_type = gr.Textbox(label="查询类型", scale=1)
            intent = gr.Textbox(label="意图分类", scale=1)
        rewrite_reason = gr.Textbox(label="改写原因", lines=2)

        gr.Markdown("#### Step 2 — Hybrid Retrieval（三路混合检索）")
        with gr.Row():
            bm25_cnt = gr.Textbox(label="BM25 召回数")
            milvus_cnt = gr.Textbox(label="Milvus 召回数")
            visual_cnt = gr.Textbox(label="图片检索数")
        with gr.Row():
            merged_cnt = gr.Textbox(label="去重后数量")
            ret_time = gr.Textbox(label="检索耗时")

        gr.Markdown("#### Step 3 — Reranker 精排")
        ranked = gr.Textbox(label="精排结果（Top N，含分数）", lines=8)

        gr.Markdown("#### Step 4 — 证据判断 + Self-RAG")
        with gr.Row():
            ev_enough = gr.Textbox(label="证据是否足够", scale=1)
            self_rag = gr.Textbox(label="Self-RAG 是否触发", scale=1)
        ev_reason = gr.Textbox(label="判断理由", lines=2)
        with gr.Row():
            suggest_q = gr.Textbox(label="Self-RAG 建议查询")
            second_cnt = gr.Textbox(label="二次检索合并数")

        gr.Markdown("#### Step 5 — Citation Verification（引用校验）")
        with gr.Row():
            citation_verified = gr.Textbox(label="校验结果", scale=1)
            citation_total_time = gr.Textbox(label="总耗时", scale=1)
        citation_details = gr.Textbox(label="逐句校验详情", lines=6)

        gr.Markdown("#### 原始召回样例")
        with gr.Row():
            bm25_sample = gr.Textbox(label="BM25 召回 Top3", lines=5)
            milvus_sample = gr.Textbox(label="Milvus 召回 Top3", lines=5)

    gr.Markdown("<div class='footer'>Powered by Qwen-Plus + Qwen3-8B · BGE-M3 Reranker · BM25 + Milvus + Visual Retrieval</div>")

    outputs = [
        answer, answer_meta, gallery,
        rewritten, query_type, intent, rewrite_reason,
        bm25_cnt, milvus_cnt, visual_cnt, merged_cnt, ret_time,
        ranked,
        ev_enough, ev_reason, self_rag,
        suggest_q, second_cnt,
        citation_verified, citation_details,
        bm25_sample, milvus_sample,
        citation_total_time,
    ]

    btn.click(fn=ask, inputs=inp, outputs=outputs)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    allowed_paths=["/root/autodl-tmp/RAG/data/saved_images"],
)
