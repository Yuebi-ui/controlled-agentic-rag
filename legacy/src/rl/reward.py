# -*- coding: utf-8 -*-
# Phase 2: 奖励函数
# 组合 citation_support_rate + 辅助信号，产生用于偏好对排序的单一奖励值

import re


def _extract_claims(raw_answer: str) -> list[dict]:
    """从含引用标记的回答中提取 claims（复用 citation_verifier 逻辑）"""
    if not raw_answer or not raw_answer.strip():
        return []
    sentences = re.split(r"(?<=[。！？\n])", raw_answer)
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        cites = re.findall(r"【(.*?)】", sent)
        cite_ids = []
        for c in cites:
            for part in re.split(r"[，,]", c):
                part = part.strip()
                if part.isdigit():
                    cite_ids.append(int(part))
        claim_text = re.sub(r"【.*?】", "", sent).strip()
        if claim_text:
            claims.append({"claim": claim_text, "cited_doc_ids": sorted(set(cite_ids))})
    return claims


def compute_reward(
    answer_text: str,
    claim_results: list[dict],
    num_context_docs: int = 5,
) -> float:
    """
    计算候选答案的综合奖励值。

    Parameters
    ----------
    answer_text : str
        原始答案文本
    claim_results : list[dict]
        CitationVerifier.verify() 返回的 claim_results，每项含 support_status
    num_context_docs : int
        上下文文档数，用于判断 invalid_citation

    Returns
    -------
    float
        奖励值，范围约 [-0.5, 1.0]，越高越好
    """
    answer_text = answer_text.strip() if answer_text else ""

    refusal_patterns = ["无答案", "无法回答", "抱歉", "无法提供", "无法确定"]

    # 1. Citation Support Rate (主信号, 0~1)
    total_claims = len(claim_results)
    if total_claims == 0:
        # 无 claim：可能是拒答或短回答
        if any(p in answer_text for p in refusal_patterns):
            return 0.5  # 中性分，依赖偏好对上下文
        else:
            # 有实质内容但无引用 → 惩罚
            return -0.3 if len(answer_text) > 20 else 0.0

    support_count = sum(
        1 for c in claim_results if c.get("support_status") == "support"
    )
    partial_count = sum(
        1 for c in claim_results if c.get("support_status") == "partial"
    )
    unsupported_count = sum(
        1 for c in claim_results
        if c.get("support_status") in ("not_support", "no_citation", "invalid_citation", "unknown")
    )

    citation_support_rate = support_count / total_claims
    unsupported_rate = unsupported_count / total_claims

    # 2. Answer length bonus（鼓励完整回答，但封顶，防刷长度）
    answer_length = len(answer_text)
    length_bonus = min(answer_length / 200.0, 0.1)  # 200 字以上给满 0.1

    # 3. 拒答惩罚（有上下文时不应轻易说拒答）
    refusal_penalty = -0.3 if any(p in answer_text for p in refusal_patterns) and num_context_docs > 0 else 0.0

    # 4. 组合（主信号占主导）
    reward = (
        0.60 * citation_support_rate   # 引用质量
        - 0.25 * unsupported_rate      # 不支持/无引用惩罚
        + 0.05 * (partial_count / max(total_claims, 1))  # 部分支持给半分
        + length_bonus                 # 长度激励（封顶 0.1）
        + refusal_penalty              # 拒答惩罚
    )

    return max(reward, -0.5)  # 下限保护
