# -*- coding: utf-8 -*-
"""用 CitationVerifier 对比两组答案的引用质量"""
import json, sys, re
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.agents.citation_verifier import CitationVerifier
from src.client.llm_judge_client import judge_client

JUDGE_MODEL_NAME = "/root/autodl-tmp/RAG/models/Qwen2.5-14B-Instruct-AWQ/"

SFT_FILE = PROJECT_DIR / "data/eval_reports/cite_answers__root_autodl-tmp_rag-server_LLaMA-Factory-main_output_qwen3_lora_sft_int4.json"
V3_FILE = PROJECT_DIR / "data/eval_reports/cite_answers_orpo-v3.json"


def compute_cite_metrics(claim_results: list[dict], total_claims: int) -> dict:
    support = sum(1 for c in claim_results if c.get("support_status") == "support")
    partial = sum(1 for c in claim_results if c.get("support_status") == "partial")
    unsupported = sum(1 for c in claim_results
                      if c.get("support_status") in ("not_support", "no_citation", "invalid_citation"))
    return {
        "total_claims": total_claims,
        "support": support,
        "partial": partial,
        "unsupported": unsupported,
        "support_rate": support / total_claims if total_claims > 0 else 0,
        "unsupported_rate": unsupported / total_claims if total_claims > 0 else 0,
        "has_citation": any(c.get("cited_doc_ids") for c in claim_results),
    }


def main():
    sft = json.load(open(SFT_FILE))
    v3 = json.load(open(V3_FILE))
    assert len(sft) == len(v3)

    verifier = CitationVerifier(judge_client, JUDGE_MODEL_NAME)

    sft_metrics = []
    v3_metrics = []
    wins = {"sft": 0, "v3": 0, "tie": 0}

    for i, (s, v) in enumerate(zip(sft, v3)):
        assert s["query"] == v["query"]
        query = s["query"]
        context = s["context"]

        # 解析上下文为 doc list（正确累积多行文档）
        docs = []
        current_doc = ""
        for line in context.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and '.' in line[:4]:
                if current_doc:
                    docs.append(current_doc)
                parts = line.split('.', 1)
                current_doc = parts[1].strip() if len(parts) > 1 else line
            else:
                current_doc += '\n' + line if current_doc else line
        if current_doc:
            docs.append(current_doc)
        if not docs:
            docs = [context]

        class FakeDoc:
            def __init__(self, c): self.page_content = c

        fake_docs = [FakeDoc(d) for d in docs]

        print(f"\n[{i+1}/{len(sft)}] {query[:60]}")

        for label, ans, store in [("SFT", s["answer"], sft_metrics), ("V3", v["answer"], v3_metrics)]:
            if not ans or len(ans) < 3:
                store.append({"total_claims": 0, "support_rate": 0, "unsupported_rate": 0,
                              "has_citation": False, "support": 0, "partial": 0, "unsupported": 0})
                print(f"  {label}: (empty)")
                continue

            result = verifier.verify(raw_answer=ans, ranked_docs=fake_docs)
            claims = result.get("claim_results", [])
            m = compute_cite_metrics(claims, len(claims))
            store.append(m)
            print(f"  {label}: claims={m['total_claims']} support={m['support']} "
                  f"partial={m['partial']} unsup={m['unsupported']} "
                  f"rate={m['support_rate']:.2f} has_cite={m['has_citation']}")

        # 判定胜负
        if sft_metrics[-1]["support_rate"] > v3_metrics[-1]["support_rate"] + 0.05:
            wins["sft"] += 1
        elif v3_metrics[-1]["support_rate"] > sft_metrics[-1]["support_rate"] + 0.05:
            wins["v3"] += 1
        else:
            wins["tie"] += 1

    # ── 汇总 ──
    def avg(metrics_list, key):
        vals = [m[key] for m in metrics_list]
        return sum(vals) / len(vals) if vals else 0

    print("\n" + "=" * 60)
    print("Citation 引用质量对比 (30 题)")
    print("=" * 60)
    print(f"{'指标':<25} {'SFT':>10} {'ORPO v3':>10}")
    print("-" * 47)
    for key, label in [("support_rate", "Citation Support Rate"),
                       ("unsupported_rate", "Unsupported Rate"),
                       ("total_claims", "Avg Claims/Answer"),
                       ("has_citation", "Has Citation %")]:
        sft_val = avg(sft_metrics, key)
        v3_val = avg(v3_metrics, key)
        if key == "has_citation":
            print(f"{label:<25} {sft_val*100:>9.1f}% {v3_val*100:>9.1f}%")
        else:
            print(f"{label:<25} {sft_val:>10.4f} {v3_val:>10.4f}")

    print(f"\n胜出统计 (gap>0.05): SFT={wins['sft']}  V3={wins['v3']}  Tie={wins['tie']}")


if __name__ == "__main__":
    main()
