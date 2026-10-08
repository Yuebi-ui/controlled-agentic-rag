# -*- coding: utf-8 -*-
# CitationVerifier：句子级引用校验，判断每个 claim 是否被 cited doc 支持

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from src.agents.prompts import CITATION_VERIFY_PROMPT


def _extract_claims(raw_answer: str) -> list[dict]:
    """从含引用标记的回答中提取 claims 及其引用编号"""
    if not raw_answer.strip():
        return []

    # 按中文句末标点分句
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
        # 清理引用标记得到纯 claim 文本
        claim_text = re.sub(r"【.*?】", "", sent).strip()
        if claim_text:
            claims.append({"claim": claim_text, "cited_doc_ids": sorted(set(cite_ids))})
    return claims


class CitationVerifier:
    """句子级引用校验器，使用 LLM Judge 判断 evidence 是否支持 claim"""

    def __init__(self, llm_client, model_name):
        self.llm_client = llm_client
        self.model_name = model_name

    def _judge_one(self, claim: str, evidence: str) -> dict:
        """对单个 claim 调用 LLM 做判断"""
        try:
            prompt = CITATION_VERIFY_PROMPT.format(claim=claim, evidence=evidence)
            completion = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.001,
                timeout=30,
            )
            raw = completion.choices[0].message.content
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            return {
                "support_status": result.get("support_status", "unknown"),
                "reason": result.get("reason", ""),
            }
        except Exception as e:
            return {"support_status": "unknown", "reason": f"LLM call failed: {e}"}

    def verify(self, raw_answer: str, ranked_docs: list[Document], query: str = "") -> dict:
        if raw_answer is None:
            return {
                "verified": True,
                "claim_results": [],
                "unsupported_count": 0,
                "partial_count": 0,
            }

        claim_results = []
        unsupported_count = 0
        partial_count = 0

        # "无答案" 直接跳过
        if "无答案" in raw_answer:
            return {
                "verified": True,
                "claim_results": [],
                "unsupported_count": 0,
                "partial_count": 0,
            }

        claims = _extract_claims(raw_answer)

        # 有事实性答案但无引用标记 → 标记为 no_citation
        if not claims or all(len(c["cited_doc_ids"]) == 0 for c in claims):
            if raw_answer.strip() and "无答案" not in raw_answer:
                return {
                    "verified": False,
                    "claim_results": [{
                        "claim": raw_answer.strip(),
                        "cited_doc_ids": [],
                        "support_status": "no_citation",
                        "reason": "答案包含事实性内容，但未提供引用标记",
                    }],
                    "unsupported_count": 1,
                    "partial_count": 0,
                }
            return {
                "verified": True,
                "claim_results": [],
                "unsupported_count": 0,
                "partial_count": 0,
            }

        # 准备需要 LLM 判决的 claim 任务
        judge_tasks = []  # (claim, cite_ids, evidence_text)
        for item in claims:
            claim = item["claim"]
            cite_ids = item["cited_doc_ids"]

            if not cite_ids:
                claim_results.append({
                    "claim": claim, "cited_doc_ids": [],
                    "support_status": "no_citation",
                    "reason": "该句未标注引用",
                })
                unsupported_count += 1
                continue

            evidence_parts = []
            valid_ids = []
            for cid in cite_ids:
                if 1 <= cid <= len(ranked_docs):
                    doc = ranked_docs[cid - 1]
                    evidence_parts.append(f"【{cid}】{doc.page_content}")
                    valid_ids.append(cid)
                else:
                    evidence_parts.append(f"【{cid}】(引用编号无效，超出文档范围)")

            if not valid_ids:
                claim_results.append({
                    "claim": claim, "cited_doc_ids": cite_ids,
                    "support_status": "invalid_citation",
                    "reason": "引用编号超出文档范围",
                })
                unsupported_count += 1
                continue

            judge_tasks.append((claim, cite_ids, "\n\n".join(evidence_parts)))

        # 并行执行 LLM 判决
        if judge_tasks:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._judge_one, claim, evidence): (claim, cite_ids)
                    for claim, cite_ids, evidence in judge_tasks
                }
                for future in as_completed(futures):
                    claim, cite_ids = futures[future]
                    judge = future.result()
                    judge["claim"] = claim
                    judge["cited_doc_ids"] = cite_ids
                    claim_results.append(judge)
                    if judge["support_status"] in ("partial", "not_support", "unknown", "invalid_citation", "no_citation"):
                        unsupported_count += 1
                        if judge["support_status"] == "partial":
                            partial_count += 1

        return {
            "verified": unsupported_count == 0,
            "claim_results": claim_results,
            "unsupported_count": unsupported_count,
            "partial_count": partial_count,
        }
