# -*- coding: utf-8 -*-
# Phase 3: 偏好对构造器
# 读取打分后的候选答案，构造 chosen/rejected 偏好对（LLaMA-Factory pairwise 格式）
# Phase 6 增强: 5 种 rejected 类型（自然分差、去引用、截断、假引用、拒答、冗余引用）

import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

# --- 配置 ---
SCORED_PATH = PROJECT_DIR / "data/rl/scored_candidates.jsonl"
OUTPUT_PATH = PROJECT_DIR / "LLaMA-Factory-main/data/rag_preference.json"
MIN_REWARD_GAP = 0.03  # chosen-rejected 最小分差（自然分差类）
LLAMA_FACTORY_DIR = PROJECT_DIR / "LLaMA-Factory-main"

SYSTEM_PROMPT = "你是特斯拉电动汽车Model 3车型的用户手册问答系统，请严格基于提供的信息回答用户问题。"
REFUSAL_TEXTS = [
    "根据提供的信息，无法确定该问题的答案。",
    "抱歉，用户手册中未包含相关信息，无法回答此问题。",
]
FAKE_CITE_IDS = [99, 98, 97]  # 假引用 ID（超出正常范围 1-8）


# ── helper: extract citation IDs from text ──────────────────────────
def _extract_cite_ids(text: str) -> list[int]:
    """提取文本中所有引用编号"""
    return [int(x) for x in re.findall(r"【(\d+)】", text)]


def _has_citations(text: str) -> bool:
    return bool(re.search(r"【\d+】", text))


# ── synthetic rejected generators ───────────────────────────────────
def _strip_citations(text: str) -> str:
    """去引用：移除所有【N】标记"""
    stripped = re.sub(r"【\d+】", "", text)
    # 清理多余空格
    stripped = re.sub(r"  +", " ", stripped)
    return stripped.strip()


def _truncate_half(text: str) -> str:
    """截断：只保留前一半句子"""
    sentences = re.split(r"(?<=[。！？\n])", text)
    half = max(1, len(sentences) // 2)
    return "".join(sentences[:half]).strip()


def _fake_citations(text: str, max_valid_id: int) -> str | None:
    """假引用：将真实引用 ID 替换为不存在的编号（99, 98, 97...）

    只在文本至少有一个有效引用时执行。
    max_valid_id: 上下文中最大的合法引用 ID
    """
    cite_ids = sorted(set(_extract_cite_ids(text)), reverse=True)
    valid_ids = [cid for cid in cite_ids if cid <= max_valid_id]
    if not valid_ids:
        return None
    result = text
    for i, vid in enumerate(valid_ids):
        fake_id = FAKE_CITE_IDS[i % len(FAKE_CITE_IDS)]
        result = result.replace(f"【{vid}】", f"【{fake_id}】")
    return result


def _redundant_citations(text: str, num_context_docs: int) -> str | None:
    """冗余引用：给已有引用的 claim 额外添加无关引用 ID

    例如原文「...速度约 30-52 km/h【1】」→「...速度约 30-52 km/h【1】,【2】,【4】」
    """
    if num_context_docs < 3:
        return None

    # 给每个已有引用追加 2 个多余的 ID（模拟过度引用）
    def _add_extra(match):
        cid = int(match.group(1))
        available = [i for i in range(1, min(num_context_docs + 1, 6)) if i != cid]
        if len(available) < 2:
            return match.group(0)
        return f"【{cid}】、【{available[0]}】、【{available[1]}】"

    result = re.sub(r"【(\d+)】", _add_extra, text)
    if result == text:
        return None
    return result


def _refusal_text() -> str:
    return REFUSAL_TEXTS[0]


# ── pair builder ────────────────────────────────────────────────────
def _make_pair(query: str, chosen_text: str, rejected_text: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": query},
        ],
        "chosen": {"from": "gpt", "value": chosen_text},
        "rejected": {"from": "gpt", "value": rejected_text},
    }


def main():
    print(f"Reading {SCORED_PATH}...")
    records = []
    with open(SCORED_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} scored records.")

    pairs = []
    stats = {
        "natural": 0,
        "strip_cite": 0,
        "truncate": 0,
        "fake_cite": 0,
        "refusal": 0,
        "redundant": 0,
        "skipped_no_cite": 0,
    }

    for record in tqdm(records, desc="Building pairs"):
        query = record["query"]
        candidates = record["candidates"]
        raw_contexts = record.get("raw_contexts", [])
        num_docs = len(raw_contexts) if raw_contexts else 5

        if len(candidates) < 2:
            continue

        # 按 reward 降序：最高分为 chosen
        sorted_cands = sorted(candidates, key=lambda c: c["score"]["reward"], reverse=True)
        chosen = sorted_cands[0]
        chosen_text = chosen["text"].strip()

        if not chosen_text or len(chosen_text) < 10 or chosen_text.startswith("[ERROR_CANDIDATE_SKIP]"):
            continue

        chosen_has_cite = _has_citations(chosen_text)
        max_valid_id = num_docs  # 最大合法引用 ID = 上下文文档数

        # ── 1. 自然分差对（最高 vs 最低 reward） ──
        if len(sorted_cands) >= 2:
            rejected_natural = sorted_cands[-1]
            gap = chosen["score"]["reward"] - rejected_natural["score"]["reward"]
            if gap >= MIN_REWARD_GAP:
                pairs.append(_make_pair(query, chosen_text, rejected_natural["text"].strip()))
                stats["natural"] += 1

        # ── 2. 去引用（chosen 有引用 → rejected 去掉引用） ──
        if chosen_has_cite:
            stripped = _strip_citations(chosen_text)
            if stripped and len(stripped) > 10 and stripped != chosen_text:
                pairs.append(_make_pair(query, chosen_text, stripped))
                stats["strip_cite"] += 1

        # ── 3. 截断（chosen 完整 → rejected 只留前半） ──
        truncated = _truncate_half(chosen_text)
        if truncated and len(truncated) > 10 and truncated != chosen_text:
            pairs.append(_make_pair(query, chosen_text, truncated))
            stats["truncate"] += 1

        # ── 4. 假引用（chosen 引真实 doc → rejected 引不存在的 ID） ──
        if chosen_has_cite:
            fake = _fake_citations(chosen_text, max_valid_id)
            if fake and fake != chosen_text and len(fake) > 10:
                pairs.append(_make_pair(query, chosen_text, fake))
                stats["fake_cite"] += 1

        # ── 5. 拒答（chosen 正常答 → rejected 说"无答案"） ──
        if "无答案" not in chosen_text and len(chosen_text) > 20:
            pairs.append(_make_pair(query, chosen_text, _refusal_text()))
            stats["refusal"] += 1

        # ── 6. 冗余引用（chosen 精确引 1 条 → rejected 引多条无关） ──
        if chosen_has_cite and num_docs >= 3:
            redundant = _redundant_citations(chosen_text, num_docs)
            if redundant and redundant != chosen_text and len(redundant) > 10:
                pairs.append(_make_pair(query, chosen_text, redundant))
                stats["redundant"] += 1

    # ── 保存 ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Breakdown:")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k}: {v}")

    # 打印抽样
    print("\n=== Sample pairs for manual review ===")
    step = max(1, len(pairs) // 5)
    for i in range(0, min(len(pairs), step * 3), step):
        p = pairs[i]
        print(f"\nQuery: {p['conversations'][1]['value'][:80]}...")
        print(f"  Chosen[:150]:   {p['chosen']['value'][:150]}...")
        print(f"  Rejected[:150]: {p['rejected']['value'][:150]}...")
        print("---")


if __name__ == "__main__":
    main()
