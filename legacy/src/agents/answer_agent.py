# -*- coding: utf-8 -*-
# AnswerAgent：基于证据生成答案

from src.agents.prompts import ANSWER_GENERATION_PROMPT, ANSWER_CORRECTION_PROMPT


class AnswerAgent:
    def __init__(self, llm_client, model_name):
        self.llm_client = llm_client
        self.model_name = model_name

    def generate(self, query: str, ranked_docs: list, stream: bool = False):
        context_parts = []
        for idx, doc in enumerate(ranked_docs):
            content = doc.page_content[:200]
            context_parts.append(f"【{idx+1}】{content}")
        context = "\n\n".join(context_parts)

        prompt = ANSWER_GENERATION_PROMPT.format(context=context, query=query)

        completion = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个有用的人工智能助手."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            frequency_penalty=0.0,
            temperature=0.1,
            top_p=0.95,
            stream=stream,
            timeout=120,
            extra_body={
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        if not stream:
            return completion.choices[0].message.content or ""

        return completion

    def rewrite_with_supported_evidence(self, query: str, original_answer: str,
                                         ranked_docs: list, verification: dict) -> str:
        """引用校验失败后，只基于被支持的证据重写答案"""
        context_parts = []
        for idx, doc in enumerate(ranked_docs):
            content = doc.page_content[:200]
            context_parts.append(f"【{idx+1}】{content}")
        context = "\n\n".join(context_parts)

        # 汇总校验发现的问题
        problems = []
        for cr in verification.get("claim_results", []):
            if cr.get("support_status") in ("partial", "not_support", "no_citation"):
                problems.append(
                    f"- {cr.get('support_status')}: {cr.get('claim', '')[:80]} | "
                    f"原因: {cr.get('reason', '')}"
                )
        verification_summary = "\n".join(problems) if problems else "部分答案未被证据支持"

        prompt = ANSWER_CORRECTION_PROMPT.format(
            context=context,
            original_answer=original_answer,
            verification_summary=verification_summary,
            query=query,
        )

        completion = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个有用的人工智能助手."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            frequency_penalty=0.0,
            temperature=0.1,
            top_p=0.95,
            timeout=120,
            extra_body={
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        return completion.choices[0].message.content or ""
