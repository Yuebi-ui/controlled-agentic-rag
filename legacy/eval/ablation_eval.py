# -*- coding: utf-8 -*-
# ablation_eval.py — 消融评测：对比不同 RAG 管道变体的效果
#
# 用法:
#   python eval/ablation_eval.py --limit 50                     # 跑前 50 条，全部变体
#   python eval/ablation_eval.py --variant hybrid_rerank         # 只跑一个变体
#   python eval/ablation_eval.py --limit 30 --variant bm25_only,agentic_rag
#
# 依赖: vLLM 服务需已启动（AnswerAgent 使用本地 vLLM）

import os
import sys
import json
import time
import hashlib
import argparse
import logging
import traceback
from text2vec import SentenceModel, semantic_search
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_tuned_model_path, text2vec_model_path
from src.utils import merge_docs

from src.client.llm_client import llm_client as remote_llm, LLM_MODEL_NAME as remote_model
from src.client.llm_local_client import llm_client as local_llm
from src.config import VLLM_MODEL_NAME as qwen3_8b_tune_model_name

from src.agents.query_agent import QueryAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.retrieval_config import get_retrieval_config
from src.agents.evidence_agent import EvidenceAgent
from src.agents.answer_agent import AnswerAgent

from eval.eval_config import (
    ABLATION_VARIANTS,
    DEFAULT_BM25_TOPK,
    DEFAULT_MILVUS_TOPK,
    DEFAULT_RERANK_TOPK,
    DEFAULT_LIMIT,
    DATA_PATH,
    OUTPUT_DIR,
    CACHE_DIR,
)
from eval.report_generator import generate as generate_report


class AblationEval:
    def __init__(self, variants: list):
        # 分析需要哪些组件
        needs = {"milvus": False, "reranker": False, "query_agent": False, "evidence_agent": False}
        for vname in variants:
            cfg = ABLATION_VARIANTS.get(vname, {})
            if cfg.get("use_milvus"): needs["milvus"] = True
            if cfg.get("use_reranker"): needs["reranker"] = True
            if cfg.get("use_query_agent"): needs["query_agent"] = True
            if cfg.get("use_evidence_agent"): needs["evidence_agent"] = True

        print("=" * 60)
        print("消融评测初始化...")

        print("  BM25 Retriever...")
        self.bm25 = BM25(docs=None, retrieve=True)

        if needs["milvus"]:
            print("  Milvus Retriever...")
            self.milvus = MilvusRetriever(docs=None, retrieve=True)
            self.milvus.retrieve_topk("测试", topk=3)
        else:
            self.milvus = None

        if needs["reranker"]:
            print("  BGE-M3 Reranker...")
            self.reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
        else:
            self.reranker = None

        print("  AnswerAgent (vLLM)...")
        self.answer_agent = AnswerAgent(llm_client=local_llm, model_name=qwen3_8b_tune_model_name)

        if needs["query_agent"] or needs["evidence_agent"]:
            print("  QueryAgent + EvidenceAgent (Remote LLM)...")
            self.query_agent = QueryAgent(llm_client=remote_llm, model_name=remote_model) if needs["query_agent"] else None
            self.evidence_agent = EvidenceAgent(llm_client=remote_llm, model_name=remote_model) if needs["evidence_agent"] else None
        else:
            self.query_agent = None
            self.evidence_agent = None

        print("  text2vec SentenceModel...")
        self.sim_model = SentenceModel(model_name_or_path=text2vec_model_path, device="cpu")

        print("初始化完成。\n")

    def run(self, variant_names: list, limit: int, output_prefix: str = "ablation"):
        data = self._load_data(limit)
        os.makedirs(CACHE_DIR, exist_ok=True)

        all_results = {}
        for vname in variant_names:
            if vname not in ABLATION_VARIANTS:
                print(f"未知变体: {vname}，跳过")
                continue
            config = ABLATION_VARIANTS[vname]
            print(f"\n{'=' * 60}")
            print(f"变体: {config['label']} ({vname})")
            print(f"{'=' * 60}")
            all_results[vname] = self._run_variant(vname, config, data)

        # 汇总报告
        output_dir = output_prefix if output_prefix != "ablation" else OUTPUT_DIR
        summary = generate_report(all_results, output_dir)
        self._print_summary(summary)
        return all_results

    def _load_data(self, limit: int) -> list:
        with open(DATA_PATH) as f:
            data = json.load(f)
        return data[:limit]

    def _run_variant(self, vname: str, config: dict, data: list) -> dict:
        cache_path = os.path.join(CACHE_DIR, f"{vname}.json")
        cached = self._load_cache(cache_path)
        results = {
            "label": config["label"],
            "questions": [],
            "pred_answers": [],
            "scores": [],
            "semantic_sims": [],
            "keyword_scores": [],
            "latencies": [],
            "per_item": [],
        }

        for i, item in enumerate(tqdm(data, desc=f"  {config['label']}", unit="q")):
            question = item["question"].strip()
            gold = item.get("answer", "")
            keywords = item.get("keywords", [])
            cache_key = hashlib.md5(f"{vname}:{question}".encode()).hexdigest()[:12]

            # 检查缓存
            if cache_key in cached:
                entry = cached[cache_key]
                results["questions"].append(question)
                results["pred_answers"].append(entry.get("pred_answer", ""))
                results["scores"].append(entry.get("score", 0))
                results["semantic_sims"].append(entry.get("semantic_sim", 0))
                results["keyword_scores"].append(entry.get("keyword_score", 0))
                results["latencies"].append(entry.get("latency", 0))
                results["per_item"].append(entry)
                continue

            t0 = time.time()

            try:
                # Step 1: Query rewrite (agentic only)
                search_query = question
                query_type = "fact_qa"
                if config["use_query_agent"]:
                    try:
                        q_info = self.query_agent.run(question)
                        search_query = q_info.get("rewritten_query", question)
                        query_type = q_info.get("query_type", "fact_qa")
                    except Exception as e:
                        print(f"\n  [WARN] QueryAgent 失败: {e}，使用原始问题")
                        search_query = question

                # Step 2: Retrieval
                if config["use_query_agent"]:
                    agentic_config = get_retrieval_config(query_type, evidence_enough=True)
                    bm25_topk = agentic_config["bm25_topk"]
                    milvus_topk = agentic_config["milvus_topk"]
                    rerank_topk = agentic_config["rerank_topk"]
                else:
                    bm25_topk = DEFAULT_BM25_TOPK
                    milvus_topk = DEFAULT_MILVUS_TOPK
                    rerank_topk = DEFAULT_RERANK_TOPK

                bm25_docs = self.bm25.retrieve_topk(search_query, topk=bm25_topk) if config["use_bm25"] else []
                milvus_docs = self.milvus.retrieve_topk(search_query, topk=milvus_topk) if config["use_milvus"] else []

                if config["use_bm25"] and config["use_milvus"]:
                    merged_docs = merge_docs(bm25_docs, milvus_docs)
                elif config["use_bm25"]:
                    merged_docs = bm25_docs
                else:
                    merged_docs = milvus_docs

                # Step 3: Rerank
                if config["use_reranker"] and merged_docs:
                    ranked_docs, ranked_scores = self.reranker.rank(
                        search_query, merged_docs, topk=min(rerank_topk, len(merged_docs)),
                        return_scores=True,
                    )
                else:
                    ranked_docs = merged_docs[:rerank_topk]
                    ranked_scores = None

                # Step 4: Evidence + Self-RAG (agentic only)
                if config["use_evidence_agent"] and config["use_self_rag"] and ranked_docs:
                    try:
                        ev_result = self.evidence_agent.judge(
                            question, query_type, ranked_docs, ranked_scores
                        )
                        if not ev_result.get("is_enough", True):
                            sc = get_retrieval_config(query_type, evidence_enough=False)
                            sq = ev_result.get("suggested_query") or search_query
                            bm25_2 = self.bm25.retrieve_topk(sq, topk=sc["bm25_topk"])
                            milvus_2 = self.milvus.retrieve_topk(sq, topk=sc["milvus_topk"])
                            all_merged = merge_docs(merged_docs, merge_docs(bm25_2, milvus_2))
                            ranked_docs, ranked_scores = self.reranker.rank(
                                question, all_merged, topk=sc["rerank_topk"], return_scores=True,
                            )
                    except Exception as e:
                        print(f"\n  [WARN] EvidenceAgent 失败: {e}")

                # Step 5: Generate answer
                if ranked_docs:
                    pred_answer = self.answer_agent.generate(question, ranked_docs, stream=False)
                else:
                    pred_answer = "无答案"

                latency = time.time() - t0

                # Step 6: Score
                score, semantic_sim, keyword_score = self._calc_score(
                    gold, pred_answer, keywords
                )

            except Exception as e:
                logger.error("问题 '%s...' 处理失败: %s\n%s", question[:40], e, traceback.format_exc())
                pred_answer = ""
                latency = time.time() - t0
                score, semantic_sim, keyword_score = 0.0, 0.0, 0.0

            # 记录
            entry = {
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "keywords": keywords,
                "score": score,
                "semantic_sim": semantic_sim,
                "keyword_score": keyword_score,
                "latency": round(latency, 2),
                "variant": vname,
            }

            results["questions"].append(question)
            results["pred_answers"].append(pred_answer)
            results["scores"].append(score)
            results["semantic_sims"].append(semantic_sim)
            results["keyword_scores"].append(keyword_score)
            results["latencies"].append(latency)
            results["per_item"].append(entry)

            # 实时写缓存
            cached[cache_key] = entry
            self._save_cache(cache_path, cached)

        return results

    def _calc_score(self, gold: str, pred: str, keywords: list) -> tuple:
        """计算综合得分，逻辑与 final_score.py 一致"""
        if gold == "无答案" and pred != gold:
            return 0.0, 0.0, 0.0
        if gold == "无答案" and pred == gold:
            return 1.0, 1.0, 1.0

        # 语义相似度（使用 semantic_search，与 final_score.py 一致）
        try:
            semantic_sim = semantic_search(
                self.sim_model.encode([gold]), self.sim_model.encode([pred]), top_k=1
            )[0][0]["score"]
        except Exception:
            semantic_sim = 0.0

        # 关键词得分
        if keywords:
            matched = [w for w in keywords if w in pred]
            keyword_score = len(matched) / len(keywords)
            score = 0.2 * keyword_score + 0.8 * semantic_sim
        else:
            keyword_score = 0.0
            score = semantic_sim

        return round(score, 4), round(semantic_sim, 4), round(keyword_score, 4)

    def _load_cache(self, path: str) -> dict:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, path: str, data: dict):
        try:
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def _print_summary(self, summary: dict):
        print(f"\n{'=' * 60}")
        print("消融评测结果汇总")
        print(f"{'=' * 60}")
        print(f"{'Variant':<30} {'Count':>6} {'Score':>8} {'SemSim':>8} {'KeyWd':>8} {'Latency':>8}")
        print("-" * 68)
        for name, s in summary.items():
            print(f"{s['label']:<30} {s['count']:>6} {s['avg_score']:>8.4f} "
                  f"{s['avg_semantic_sim']:>8.4f} {s['avg_keyword_score']:>8.4f} "
                  f"{s['avg_latency']:>7.2f}s")
        print(f"\n报告已保存到: {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="RAG Ablation Evaluation")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"评测问题数量上限 (default: {DEFAULT_LIMIT})")
    parser.add_argument("--variant", type=str, default="",
                        help="只跑指定变体，逗号分隔 (e.g. hybrid_rerank,agentic_rag)")
    parser.add_argument("--output", type=str, default="ablation",
                        help="输出文件前缀 (default: ablation)")
    args = parser.parse_args()

    if args.variant:
        variants = [v.strip() for v in args.variant.split(",")]
    else:
        variants = list(ABLATION_VARIANTS.keys())

    print(f"评测变体: {', '.join(variants)}")
    print(f"问题上限: {args.limit}")

    evaluator = AblationEval(variants)
    evaluator.run(variants, args.limit, args.output)


if __name__ == "__main__":
    main()
