# -*- coding: utf-8 -*-
# Phase 2: 候选答案打分器
# 使用本地 Citation Judge (Qwen2.5-14B vLLM :8001) 对每个候选答案的引用质量打分

import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.agents.citation_verifier import CitationVerifier
from src.client.llm_judge_client import judge_client, JUDGE_MODEL_NAME
from src.rl.reward import compute_reward

# --- 配置 ---
CANDIDATES_PATH = PROJECT_DIR / "data/rl/candidates.jsonl"
OUTPUT_PATH = PROJECT_DIR / "data/rl/scored_candidates.jsonl"
MAX_WORKERS = 4  # 并发调用 judge


def score_one_candidate(verifier: CitationVerifier, raw_answer: str, ranked_docs: list) -> dict:
    """对单个候选答案打分，返回 citation 校验结果 + reward"""
    # 模拟 Document 对象（CitationVerifier 需要 .page_content 属性）
    class FakeDoc:
        def __init__(self, content):
            self.page_content = content
    docs = [FakeDoc(d) for d in ranked_docs]

    verify_result = verifier.verify(raw_answer=raw_answer, ranked_docs=docs)
    claim_results = verify_result.get("claim_results", [])
    reward = compute_reward(
        answer_text=raw_answer,
        claim_results=claim_results,
        num_context_docs=len(ranked_docs),
    )
    return {
        "claim_results": claim_results,
        "unsupported_count": verify_result.get("unsupported_count", 0),
        "partial_count": verify_result.get("partial_count", 0),
        "verified": verify_result.get("verified", False),
        "reward": reward,
    }


def main():
    # 读取候选答案
    print(f"Reading {CANDIDATES_PATH}...")
    records = []
    with open(CANDIDATES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records.")

    # 初始化 CitationVerifier（使用本地 judge）
    verifier = CitationVerifier(judge_client, JUDGE_MODEL_NAME)

    # 对所有候选答案打分（串行处理 candidates，CitationVerifier 内部有 4-worker 并行）
    print(f"Scoring {len(records)} x ~4 candidates with judge...")
    scored_records = []
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for record in tqdm(records, desc="Scoring"):
            query = record["query"]
            # 兼容：部分旧记录没有 raw_contexts，从 context 字符串回退
            raw_contexts = record.get("raw_contexts")
            if raw_contexts is None:
                context_str = record.get("context", "")
                raw_contexts = [
                    line.split(".", 1)[-1].strip()
                    for line in context_str.split("\n")
                    if line.strip() and "." in line[:4]
                ]
            if not raw_contexts:
                raw_contexts = [record.get("context", "")]
            scored_candidates = []

            for cand in record["candidates"]:
                try:
                    score = score_one_candidate(verifier, cand["text"], raw_contexts)
                except Exception as e:
                    score = {
                        "claim_results": [],
                        "unsupported_count": 0,
                        "partial_count": 0,
                        "verified": False,
                        "reward": -0.5,
                        "error": str(e),
                    }
                scored_candidates.append({
                    "temperature": cand["temperature"],
                    "text": cand["text"],
                    "score": score,
                })

            out_record = {
                "query": query,
                "raw_contexts": raw_contexts,
                "candidates": scored_candidates,
            }
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out.flush()
            scored_records.append(out_record)

    print(f"\nDone! Scored {len(scored_records)} records.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
