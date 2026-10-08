# -*- coding: utf-8 -*-
"""Citation 专项评测：对比两个模型的引用质量"""
import json, sys, os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from openai import OpenAI
from src.agents.prompts import ANSWER_GENERATION_PROMPT

VLLM_URL = "http://localhost:8000/v1"
OUTPUT_DIR = PROJECT_DIR / "data/eval_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 从 candidates.jsonl 取前 30 条作为固定测试集
CANDIDATES_PATH = PROJECT_DIR / "data/rl/candidates.jsonl"
N_SAMPLES = 30


def load_queries():
    queries = []
    with open(CANDIDATES_PATH) as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLES:
                break
            r = json.loads(line)
            queries.append({"query": r["query"], "context": r["context"]})
    return queries


def generate_answers(model_name: str, queries: list, output_file: str):
    """用指定模型生成答案"""
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
    results = []
    for i, q in enumerate(queries):
        prompt = ANSWER_GENERATION_PROMPT.format(context=q["context"], query=q["query"])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个有用的人工智能助手."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
            top_p=0.95,
            extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = resp.choices[0].message.content.strip()
        results.append({"query": q["query"], "context": q["context"], "answer": answer})
        print(f"  [{i+1}/{len(queries)}] {q['query'][:50]}... -> {len(answer)} chars")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} answers to {output_file}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else os.getenv("VLLM_MODEL_NAME", "sft")
    out = str(OUTPUT_DIR / f"cite_answers_{model.replace('/', '_')}.json")

    queries = load_queries()
    print(f"Generating {len(queries)} answers with model={model}")
    generate_answers(model, queries, out)
