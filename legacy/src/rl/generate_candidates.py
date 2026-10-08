# -*- coding: utf-8 -*-
# Phase 1: 候选答案生成器
# 读取已有 train_data.json，用 SFT 模型生成多温度候选答案

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 确保项目根目录在 path 中
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from openai import OpenAI
from src.agents.prompts import ANSWER_GENERATION_PROMPT

# --- 配置 ---
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "/root/autodl-tmp/rag-server/LLaMA-Factory-main/output/qwen3_lora_sft_int4"

# Phase 6.2: 多样化采样配置（温度 + top_p + top_k 组合）
# 关键修复：去掉 top_k=1（强制贪婪），让 temperature 真正生效
SAMPLING_CONFIGS = [
    {"temperature": 0.5, "top_p": 0.9},                     # 保守
    {"temperature": 1.0, "top_p": 0.95, "top_k": 50},       # 适中 + top_k 约束
    {"temperature": 1.2, "top_p": 0.85},                     # 激进
    {"temperature": 1.5, "top_p": 0.80},                     # 高多样性
]
MAX_TOKENS = 1024
MAX_SAMPLES = 300

TRAIN_DATA_PATH = PROJECT_DIR / "data/qa_pairs/train_data.json"
OUTPUT_DIR = PROJECT_DIR / "data/rl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "candidates.jsonl"

# --- 初始化客户端 ---
client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)


def _gen_one_candidate(args: tuple) -> tuple:
    """生成一个采样配置下的答案（用于线程池）"""
    query, context, config = args
    prompt = ANSWER_GENERATION_PROMPT.format(context=context, query=query)
    temp = config["temperature"]
    top_p = config.get("top_p", 0.95)
    top_k = config.get("top_k", None)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    if top_k is not None:
        extra_body["top_k"] = top_k
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=temp,
                top_p=top_p,
                timeout=120,
                extra_body=extra_body,
            )
            return (temp, completion.choices[0].message.content.strip())
        except Exception as e:
            if attempt == 2:
                return (temp, f"[ERROR_CANDIDATE_SKIP] {e}")
            time.sleep(3)


def generate_candidates_parallel(query: str, context: str) -> list[dict]:
    """并行生成多个采样配置下的候选答案"""
    tasks = [(query, context, cfg) for cfg in SAMPLING_CONFIGS]
    results = {}
    with ThreadPoolExecutor(max_workers=len(SAMPLING_CONFIGS)) as ex:
        futures = {ex.submit(_gen_one_candidate, task): task[2]["temperature"] for task in tasks}
        for fut in as_completed(futures):
            temp, text = fut.result()
            results[temp] = text
    candidates = []
    for cfg in SAMPLING_CONFIGS:
        t = cfg["temperature"]
        candidates.append({
            "temperature": t,
            "top_p": cfg.get("top_p", 0.95),
            "top_k": cfg.get("top_k"),
            "text": results.get(t, "[MISSING]"),
        })
    return candidates


def main():
    print(f"Reading {TRAIN_DATA_PATH}...")
    with open(TRAIN_DATA_PATH) as f:
        lines = f.readlines()

    items = []
    for line in lines:
        info = json.loads(line)
        query = info["query"].strip()
        context_list = info.get("context", [])
        if not query or not context_list:
            continue
        context = "\n".join(
            [f"{idx + 1}.{doc}" for idx, doc in enumerate(context_list)]
        )
        items.append({"query": query, "context": context, "raw_contexts": context_list})

    items = items[:MAX_SAMPLES]
    cfgs_desc = ", ".join(f"T={c['temperature']}" for c in SAMPLING_CONFIGS)
    print(f"Loaded {len(items)} queries. Generating {len(SAMPLING_CONFIGS)} candidates each ({cfgs_desc})...")

    # 断点续跑
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            start_idx = sum(1 for _ in f)
        if start_idx > 0:
            print(f"Resuming from record {start_idx}")

    with open(OUTPUT_PATH, "a" if start_idx > 0 else "w", encoding="utf-8") as out:
        for i in tqdm(range(start_idx, len(items)), desc="Generating", initial=start_idx, total=len(items)):
            item = items[i]
            candidates = generate_candidates_parallel(item["query"], item["context"])
            record = {
                "query": item["query"],
                "context": item["context"],
                "raw_contexts": item["raw_contexts"],
                "candidates": candidates,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nDone! {len(items)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
