# -*- coding: utf-8 -*-
# no_answer_eval.py — 无答案专项评测：评测系统拒答能力
#
# 用法:
#   python eval/no_answer_eval.py                     # 默认 50 条有答案 + 全部无答案
#   python eval/no_answer_eval.py --has-answer-limit 20
#   python eval/no_answer_eval.py --output my_report
#
# 指标:
#   No-answer Precision: 系统说无答案时，真的应该无答案的比例
#   No-answer Recall:   应该无答案的问题中，系统成功拒答的比例
#   Hallucination Rate: 无答案问题中系统编造答案的比例
#   False Refusal Rate: 有答案问题中系统错误拒答的比例

import os
import sys
import json
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs

from src.client.llm_local_client import llm_client as local_llm
from src.config import VLLM_MODEL_NAME as qwen3_8b_tune_model_name
from src.agents.answer_agent import AnswerAgent

DATA_PATH = "data/qa_pairs/test_qa_pair_verify.json"
NO_ANSWER_DATA_PATH = "data/qa_pairs/no_answer_qa.json"
OUTPUT_DIR = "data/eval_reports"


def is_refused(pred_answer: str) -> bool:
    """判断系统是否拒答（匹配"无答案"字面 或 礼貌拒答模式）"""
    refusal_patterns = [
        "无答案",
        "无法回答",
        "无法提供相关信息",
        "根据提供的信息，无法确定",
        "抱歉，我是特斯拉",
        "抱歉，无法",
        "无法执行该操作",
        "只能回答与",
        "没有相关信息",
    ]
    return any(p in pred_answer for p in refusal_patterns)


def load_data(has_answer_limit: int) -> tuple:
    """加载有答案和无答案数据"""
    with open(DATA_PATH) as f:
        all_data = json.load(f)

    # 有答案样本
    has_answer = [d for d in all_data if d.get("answer") != "无答案"]
    has_answer = has_answer[:has_answer_limit]

    # 无答案样本（来自两个数据集）
    with open(NO_ANSWER_DATA_PATH) as f:
        no_answer_data = json.load(f)

    # test_qa_pair_verify.json 中已有的无答案样本
    existing_no_answer = [d for d in all_data if d.get("answer") == "无答案"]

    return has_answer, no_answer_data, existing_no_answer


def run_pipeline(questions: list, bm25, milvus, reranker, answer_agent,
                 bm25_topk=5, milvus_topk=10, rerank_topk=5):
    """对一批问题跑管道，返回预测结果列表"""
    preds = []
    for item in tqdm(questions, desc="  评测中", unit="q"):
        query = item["question"].strip()

        try:
            # 检索
            bm25_docs = bm25.retrieve_topk(query, topk=bm25_topk)
            milvus_docs = milvus.retrieve_topk(query, topk=milvus_topk)
            merged = merge_docs(bm25_docs, milvus_docs)

            # 精排
            ranked = reranker.rank(query, merged, topk=min(rerank_topk, len(merged)))

            # 生成
            answer = answer_agent.generate(query, ranked, stream=False)
        except Exception as e:
            answer = f"<ERROR: {e}>"

        preds.append(answer)
    return preds


def evaluate(has_answer_items, has_answer_preds,
             no_answer_items, no_answer_preds) -> dict:
    """计算四项核心指标"""
    # FP: 有答案但系统拒答
    fp = sum(1 for pred in has_answer_preds if is_refused(pred))
    total_has = len(has_answer_preds)
    false_refusal_rate = fp / total_has if total_has > 0 else 0.0

    # TP + FN: 无答案问题
    total_no = len(no_answer_preds)
    tp = sum(1 for pred in no_answer_preds if is_refused(pred))
    fn = total_no - tp  # 该拒答却回答了

    no_answer_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    no_answer_recall = tp / total_no if total_no > 0 else 0.0
    hallucination_rate = fn / total_no if total_no > 0 else 0.0

    return {
        "no_answer_precision": round(no_answer_precision, 4),
        "no_answer_recall": round(no_answer_recall, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_no_answer": total_no,
        "total_has_answer": total_has,
    }


def breakdown_by_type(items: list, preds: list) -> dict:
    """按类型统计拒答率"""
    by_type = {}
    for item, pred in zip(items, preds):
        t = item.get("type", "unknown")
        if t not in by_type:
            by_type[t] = {"count": 0, "refused": 0, "answered": 0}
        by_type[t]["count"] += 1
        if is_refused(pred):
            by_type[t]["refused"] += 1
        else:
            by_type[t]["answered"] += 1

    for t in by_type:
        c = by_type[t]["count"]
        by_type[t]["refusal_rate"] = round(by_type[t]["refused"] / c, 4) if c > 0 else 0.0

    return by_type


def generate_report(metrics: dict, by_type: dict,
                    hallucination_examples: list, false_refusal_examples: list,
                    output_prefix: str = "no_answer"):
    """生成 JSON 和 Markdown 报告"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_result.json")
    result = {
        "metrics": metrics,
        "breakdown_by_type": by_type,
        "hallucination_examples": hallucination_examples,
        "false_refusal_examples": false_refusal_examples,
    }
    with open(json_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Markdown
    md_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_report.md")
    lines = [
        "# No-Answer Evaluation Report",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| No-Answer Precision（拒答精确率） | {metrics['no_answer_precision']:.2%} |",
        f"| No-Answer Recall（拒答召回率） | {metrics['no_answer_recall']:.2%} |",
        f"| Hallucination Rate（幻觉率） | {metrics['hallucination_rate']:.2%} |",
        f"| False Refusal Rate（错误拒答率） | {metrics['false_refusal_rate']:.2%} |",
        "",
        f"参考值: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}",
        f"  (无答案 {metrics['total_no_answer']} 条, 有答案 {metrics['total_has_answer']} 条)",
        "",
        "## Breakdown by Type",
        "",
        "| Type | Count | Refused | Answered | Refusal Rate |",
        "|------|-------|---------|----------|-------------|",
    ]
    for t, info in sorted(by_type.items()):
        lines.append(
            f"| {t} | {info['count']} | {info['refused']} "
            f"| {info['answered']} | {info['refusal_rate']:.1%} |"
        )

    if hallucination_examples:
        lines += ["", "## Hallucination Examples（该拒答却回答了）", ""]
        for i, ex in enumerate(hallucination_examples[:5]):
            lines.append(f"**{i+1}.** Q: {ex['question']}")
            lines.append(f"> A: {ex['pred'][:150]}...")
            lines.append("")

    if false_refusal_examples:
        lines += ["", "## False Refusal Examples（该回答却拒答了）", ""]
        for i, ex in enumerate(false_refusal_examples[:5]):
            lines.append(f"**{i+1}.** Q: {ex['question']}")
            lines.append(f"> Gold: {ex['gold'][:100]}...")
            lines.append(f"> Pred: {ex['pred']}")
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="No-Answer Evaluation")
    parser.add_argument("--has-answer-limit", type=int, default=50,
                        help="有答案样本数量 (default: 50)")
    parser.add_argument("--output", type=str, default="no_answer",
                        help="输出文件前缀 (default: no_answer)")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "agentic"],
                        help="评测模式: baseline (BM25+Milvus+Reranker) 或 agentic (完整 RAGPipeline)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"无答案专项评测 (模式: {args.mode})")

    if args.mode == "agentic":
        # Agentic 模式：使用完整 RAGPipeline
        from src.pipeline.rag_pipeline import RAGPipeline
        print("\n初始化 RAGPipeline (QueryAgent + EvidenceAgent + Self-RAG)...")
        pipeline = RAGPipeline()
        answer_agent = None

        def run_agentic(questions: list) -> list:
            preds = []
            for item in tqdm(questions, desc="  评测中 (agentic)", unit="q"):
                query = item["question"].strip()
                try:
                    result = pipeline.run(query)
                    preds.append(result["answer"])
                except Exception as e:
                    preds.append(f"<ERROR: {e}>")
            return preds
    else:
        # Baseline 模式：原始逻辑
        print("\n初始化组件...")
        print("  BM25 + Milvus...")
        bm25 = BM25(docs=None, retrieve=True)
        milvus = MilvusRetriever(docs=None, retrieve=True)
        milvus.retrieve_topk("测试", topk=3)
        print("  Reranker + AnswerAgent...")
        reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
        answer_agent = AnswerAgent(llm_client=local_llm, model_name=qwen3_8b_tune_model_name)
        pipeline = None

        def run_agentic(questions: list) -> list:
            return run_pipeline(questions, bm25, milvus, reranker, answer_agent)

    # 加载数据
    has_answer, no_answer_new, no_answer_existing = load_data(args.has_answer_limit)
    all_no_answer = no_answer_new + no_answer_existing
    print(f"\n数据: 有答案 {len(has_answer)} 条, 无答案 {len(all_no_answer)} 条")
    print(f"  新增无答案: {len(no_answer_new)} 条, 来自 test_qa_pair: {len(no_answer_existing)} 条")

    # 跑有答案样本
    print("\n--- 有答案样本 ---")
    t0 = time.time()
    has_answer_preds = run_agentic(has_answer)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 跑无答案样本
    print("\n--- 无答案样本 ---")
    t0 = time.time()
    no_answer_preds = run_agentic(all_no_answer)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 计算指标
    metrics = evaluate(has_answer, has_answer_preds, all_no_answer, no_answer_preds)
    by_type = breakdown_by_type(no_answer_new, no_answer_preds[:len(no_answer_new)])

    # 收集例子
    hallucination_examples = [
        {"question": all_no_answer[i]["question"],
         "pred": no_answer_preds[i],
         "type": all_no_answer[i].get("type", "")}
        for i in range(len(all_no_answer))
        if not is_refused(no_answer_preds[i])
    ]
    false_refusal_examples = [
        {"question": has_answer[i]["question"],
         "gold": has_answer[i]["answer"],
         "pred": has_answer_preds[i]}
        for i in range(len(has_answer))
        if is_refused(has_answer_preds[i])
    ]

    metrics["mode"] = args.mode

    # 输出报告
    json_path, md_path = generate_report(
        metrics, by_type,
        hallucination_examples, false_refusal_examples,
        output_prefix=f"{args.output}_{args.mode}",
    )

    # 终端汇总
    print(f"\n{'=' * 60}")
    print(f"评测结果 ({args.mode})")
    print(f"{'=' * 60}")
    print(f"  拒答精确率 (No-Answer Precision):  {metrics['no_answer_precision']:.2%}")
    print(f"  拒答召回率 (No-Answer Recall):     {metrics['no_answer_recall']:.2%}")
    print(f"  幻觉率 (Hallucination Rate):       {metrics['hallucination_rate']:.2%}")
    print(f"  错误拒答率 (False Refusal Rate):   {metrics['false_refusal_rate']:.2%}")
    print(f"\n  幻觉案例数: {len(hallucination_examples)}")
    print(f"  错误拒答案例数: {len(false_refusal_examples)}")
    print(f"\n报告已保存:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")


if __name__ == "__main__":
    main()
