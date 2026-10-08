# -*- coding: utf-8 -*-
# QueryAgent：问题改写 + 分类

import json
import time
import logging
from src.agents.prompts import QUERY_REWRITE_PROMPT
from src.agents.query_schema import QueryRewriteResult

logger = logging.getLogger(__name__)


class QueryAgent:
    def __init__(self, llm_client, model_name, max_retries=1):
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_retries = max_retries

    def _parse_response(self, raw: str, query: str) -> dict:
        """解析 LLM 返回的 JSON，失败时返回 None"""
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            result = json.loads(raw)
            QueryRewriteResult(**result)
            return result
        except Exception:
            return None

    def run(self, query: str) -> dict:
        fallback = {
            "original_query": query,
            "rewritten_query": query,
            "need_retrieval": True,
            "reasoning": "LLM 调用失败，使用原问题",
            "query_type": "fact_qa",
            "intent": "knowledge_qa",
        }

        for attempt in range(self.max_retries + 1):
            try:
                prompt = QUERY_REWRITE_PROMPT.format(query=query)
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.001,
                    timeout=30,
                )
                raw = completion.choices[0].message.content
                result = self._parse_response(raw, query)
                if result is not None:
                    result.setdefault("original_query", query)
                    result.setdefault("rewritten_query", query)
                    result.setdefault("need_retrieval", True)
                    result.setdefault("reasoning", "")
                    result.setdefault("query_type", "fact_qa")
                    result.setdefault("intent", "knowledge_qa")
                    return result
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning("Attempt %d failed: %s, retrying in %ds", attempt + 1, e, 2 ** attempt)
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("LLM call failed: %s, using fallback", e)

        return fallback
