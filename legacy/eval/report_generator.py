# -*- coding: utf-8 -*-
# 报告生成器：将消融评测结果输出为 JSON / CSV / Markdown

import json
import csv
import os
import numpy as np


def generate(results: dict, output_dir: str, prefix: str = "ablation") -> dict:
    """生成三种格式的评测报告。

    Args:
        results: {variant_name: {"scores": [...], "semantic_sims": [...],
                  "keyword_scores": [...], "latencies": [...], "count": N}}
        output_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        summary: 汇总统计数据
    """
    os.makedirs(output_dir, exist_ok=True)

    # 计算汇总统计
    summary = {}
    for name, data in results.items():
        scores = data.get("scores", [])
        semantic_sims = data.get("semantic_sims", [])
        keyword_scores = data.get("keyword_scores", [])
        latencies = data.get("latencies", [])

        summary[name] = {
            "label": data.get("label", name),
            "count": len(scores),
            "avg_score": round(np.mean(scores), 4) if scores else 0,
            "avg_semantic_sim": round(np.mean(semantic_sims), 4) if semantic_sims else 0,
            "avg_keyword_score": round(np.mean(keyword_scores), 4) if keyword_scores else 0,
            "avg_latency": round(np.mean(latencies), 2) if latencies else 0,
        }

    # 1. JSON
    json_path = os.path.join(output_dir, f"{prefix}_result.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    # 2. CSV — 每条 question 的得分矩阵
    csv_path = os.path.join(output_dir, f"{prefix}_result.csv")
    variant_names = list(results.keys())
    if variant_names:
        first_variant = results[variant_names[0]]
        num_questions = len(first_variant.get("scores", []))
        questions = first_variant.get("questions", [str(i) for i in range(num_questions)])

        with open(csv_path, "w", newline="") as f:
            fieldnames = ["index", "question"] + [f"{v}_score" for v in variant_names]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(num_questions):
                row = {"index": i, "question": questions[i] if i < len(questions) else ""}
                for v in variant_names:
                    scores = results[v].get("scores", [])
                    row[f"{v}_score"] = scores[i] if i < len(scores) else ""
                writer.writerow(row)

    # 3. Markdown
    md_path = os.path.join(output_dir, f"{prefix}_report.md")
    _write_markdown(summary, results, md_path)

    return summary


def _write_markdown(summary: dict, results: dict, path: str):
    variant_names = list(summary.keys())

    lines = [
        "# RAG Ablation Report",
        "",
        "## Overall Metrics",
        "",
        "| Variant | Count | Avg Score | Semantic Sim | Keyword Score | Avg Latency |",
        "|---------|-------|-----------|-------------|---------------|-------------|",
    ]

    for name, s in summary.items():
        lines.append(
            f"| {s['label']} | {s['count']} | {s['avg_score']} | {s['avg_semantic_sim']} "
            f"| {s['avg_keyword_score']} | {s['avg_latency']}s |"
        )

    # 逐级增益
    lines += [
        "",
        "## Per-Component Gain",
        "",
    ]

    gains = _calc_gains(summary, variant_names)
    for g in gains:
        lines.append(f"- **{g['label']}**: {g['delta']:+.4f} ({g['from']} → {g['to']})")

    # 各变体得分分布（min/max/std）
    lines += [
        "",
        "## Score Distribution",
        "",
        "| Variant | Min | Max | Std |",
        "|---------|-----|-----|-----|",
    ]
    for name in variant_names:
        scores = results[name].get("scores", [])
        if scores:
            lines.append(
                f"| {summary[name]['label']} | {min(scores):.4f} | {max(scores):.4f} | {np.std(scores):.4f} |"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _calc_gains(summary: dict, variant_names: list) -> list:
    """计算逐级增益"""
    gains = []
    pairs = [
        ("milvus_only", "bm25_only", "Milvus vs BM25"),
        ("hybrid", "bm25_only", "Hybrid vs BM25"),
        ("hybrid_rerank", "hybrid", "Reranker gain"),
        ("agentic_rag", "hybrid_rerank", "Agentic gain (QueryRewrite + Evidence + Self-RAG)"),
    ]
    for a, b, label in pairs:
        if a in summary and b in summary:
            delta = summary[a]["avg_score"] - summary[b]["avg_score"]
            gains.append({
                "label": label,
                "delta": round(delta, 4),
                "from": summary[b]["avg_score"],
                "to": summary[a]["avg_score"],
            })
    return gains
