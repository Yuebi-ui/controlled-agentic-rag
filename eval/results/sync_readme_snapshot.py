"""Synchronize README metrics from the maintainer-reported published record bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


START = "## Published run — maintainer-reported records"
END = "## Run locally"


def sync(repo: Path, run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    b = summary["end_to_end"]["variants"]["basic_rag"]
    f = summary["end_to_end"]["variants"]["controlled_full"]
    r = summary["retrieval"]
    rt = summary["router"]
    rd = summary["round2"]
    g = summary["generation"]
    paired = summary["end_to_end"]["paired"]["grounded_success_per_input"]

    block = f"""## Published run — maintainer-reported records

> **来源与认证范围**：维护者声明 `{summary['run_id']}` 下的 {summary['query_count']} 条 Query 记录来自真实实验（Gold 可回答 {summary['answerable_count']}、不可回答 {summary['unanswerable_count']}）。本仓库能从逐条 JSONL 复算下列指标，但未包含可独立验证的模型执行、Gold 标注/人工复核或真实 GPU 采样原始日志；`real_audited` 是维护者提供的来源标记，并非第三方审计认证。

| 指标 | 对照 | 完整方案 | 说明 |
|---|---:|---:|---|
| Grounded success / input | {b['grounded_success_per_input']*100:.1f}% | **{f['grounded_success_per_input']*100:.1f}%** | {summary['query_count']} 个配对问题 |
| Correct answer / answerable | {b['correct_answer_per_answerable']*100:.1f}% | **{f['correct_answer_per_answerable']*100:.1f}%** | {summary['answerable_count']} 个可回答问题 |
| Child Recall@10 | {r['child_recall_at_10']['bm25']*100:.1f}% | **{r['child_recall_at_10']['hybrid']*100:.1f}%** | BM25 vs Hybrid |
| Reranker NDCG@5 | {r['reranker_ndcg_at_5']['before']:.3f} | **{r['reranker_ndcg_at_5']['after']:.3f}** | 相同候选集合 |
| Evidence Router Macro-F1 | {rt['rule']['macro_f1']:.3f} | **{rt['learned']['macro_f1']:.3f}** | Gold 三状态 |
| Round2 Rescue (eligible) | — | **{rd['rescue_rate_eligible']*100:.1f}%** | 仅触发且 Gold 可回答案例 |
| DPO faithfulness (judged answers) | {g['sft']['faithfulness_given_answered']*100:.1f}% (SFT) | **{g['dpo']['faithfulness_given_answered']*100:.1f}%** | 双 reviewer 标签（人工来源未独立核验） |
| Unanswerable false-answer rate ↓ | {b['unsafe_answer_per_unanswerable']*100:.1f}% | **{f['unsafe_answer_per_unanswerable']*100:.1f}%** | {summary['unanswerable_count']} 个不可回答问题 |
| Final-gate violations | — | **{rd['final_gate_violation_count']}** | answered 必须经过 sufficient gate |
| P95 E2E latency ↓ | {b['p95_latency_ms']/1000:.2f}s | **{f['p95_latency_ms']/1000:.2f}s** | 工作流记录中的延迟，不等于 GPU 硬件监测 |

Grounded Success 的配对差值为 **{paired['delta_full_minus_basic']*100:+.1f} pp**，Bootstrap 95% CI **[{paired['bootstrap_ci_95'][0]*100:+.1f}, {paired['bootstrap_ci_95'][1]*100:+.1f}] pp**（基于公开记录的统计不确定性；不验证数据产生过程）。

**记录与校验：** [Published-run 报告](docs/experiments/published_runs/RUN-EXAMPLE-2026-001/RESULTS.md) · [Manifest](docs/experiments/published_runs/RUN-EXAMPLE-2026-001/manifest.json) · [实验说明](docs/experiments/README.md) · [指标定义](eval/data/SCHEMAS.md)

"""
    path = repo / "README.md"
    text = path.read_text(encoding="utf-8")
    left = text.index(START)
    right = text.index(END)
    path.write_text(text[:left] + block + text[right:], encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, default=Path(
        "docs/experiments/published_runs/RUN-EXAMPLE-2026-001"))
    args = parser.parse_args()
    sync(args.repo, args.run)


if __name__ == "__main__":
    main()
