# -*- coding: utf-8 -*-
# EvidenceAgent：判断证据是否足够支撑答案

import json
import logging
from src.agents.prompts import EVIDENCE_JUDGE_PROMPT

logger = logging.getLogger(__name__)


class EvidenceAgent:
    def __init__(self, llm_client, model_name):
        self.llm_client = llm_client
        self.model_name = model_name

    def judge(self, query: str, query_type: str, ranked_docs: list, scores: list = None) -> dict:
        fallback = {
            "is_enough": len(ranked_docs) > 0,
            "reason": "LLM 调用失败，根据文档数量自动判断",
            "suggested_query": "",
        }

        try:
            # 构造文档摘要（截断每条文档避免超长）
            doc_summaries = []
            for i, doc in enumerate(ranked_docs):
                score_info = f" [相关度: {scores[i]:.4f}]" if scores else ""
                content = doc.page_content[:300].replace("\n", " ")
                doc_summaries.append(f"【{i+1}】{score_info}\n{content}...")

            docs_text = "\n\n".join(doc_summaries)
            prompt = EVIDENCE_JUDGE_PROMPT.format(
                query=query,
                query_type=query_type,
                ranked_docs=docs_text,
            )

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
            result.setdefault("is_enough", len(ranked_docs) > 0)
            result.setdefault("reason", "")
            result.setdefault("suggested_query", "")
            return result

        except Exception as e:
            logger.warning("调用失败: %s，使用 fallback", e)
            return fallback
