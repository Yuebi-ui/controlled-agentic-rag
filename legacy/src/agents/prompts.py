# -*- coding: utf-8 -*-
# Agent Prompt 模板集中管理

# QueryAgent：问题改写 + 分类
QUERY_REWRITE_PROMPT = """你是一个RAG检索系统的查询优化器。请对用户问题进行改写和分类。

### 改写规则
1. 口语化表达改为正式书面语（如"这车"改为"Model 3"）
2. 指代不明确的地方具体化（如"怎么关"补充为"如何关闭某功能"）
3. 如果原问题已经很清晰，保持原样
4. 不要改变用户意图

### 分类规则
- fact_qa: 事实类问答（询问某个功能、参数、操作方法）
- compare: 对比类问题（需要比较两个或多个事物）
- summary: 总结类问题（需要概括一段内容）
- multi_hop: 多跳推理问题（需要关联多个知识点才能回答）
- other: 其他类型

### 意图识别规则
你需要同时判断用户意图，分为以下4类：
- knowledge_qa: 用户询问知识库可以回答的问题（如"Model 3的续航是多少"）
- chitchat: 闲聊、问候、寒暄（如"你好"、"今天天气怎么样"）
- action_request: 请求执行某个操作，非知识问答（如"帮我写代码"、"帮我画画"）
- ambiguous: 意图模糊，无法判断用户想做什么

### 用户问题
{query}

### 输出格式（严格JSON，不要输出其他内容）
{{
  "original_query": "原始问题",
  "rewritten_query": "改写后的问题",
  "need_retrieval": true,
  "reasoning": "改写理由",
  "query_type": "fact_qa",
  "intent": "knowledge_qa"
}}"""


# EvidenceAgent：证据充分性判断
EVIDENCE_JUDGE_PROMPT = """你是一个RAG系统的证据评估器。请判断当前召回的文档是否足以回答用户问题。

### 用户问题
{query}

### 问题类型
{query_type}

### 召回的文档（按相关性从高到低排列）
{ranked_docs}

### 判断标准
1. 文档内容是否与问题直接相关
2. 文档是否覆盖了问题的所有关键方面
3. 对比类问题是否双方都有覆盖
4. 多跳问题是否所有子问题都有覆盖
5. 前3条文档的标题/内容是否能支撑答案

### 输出格式（严格JSON）
{{
  "is_enough": true,
  "reason": "判断理由",
  "suggested_query": "如果证据不足，给出更精准的检索词；如果证据足够，留空"
}}"""


# AnswerAgent：答案生成（句子级引用约束）
ANSWER_GENERATION_PROMPT = """### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统，请严格基于【信息】中的内容回答用户问题。

### 要求
1. 答案需要精准、通顺，用中文回答
2. 每个关键事实句后标注引用编号，格式：事实内容【编号】
3. 如果某个陈述在【信息】中找不到原文支持，不要编造
4. 如果无法从【信息】中得到答案，输出"无答案"

### 问题
{query}

### 回答
"""


# CitationVerifier：引用一致性校验
CITATION_VERIFY_PROMPT = """你是一个 RAG 答案引用校验器。请判断给定 evidence 是否能够支持 claim。

### 要求
1. 如果 evidence 明确支持 claim，输出 support。
2. 如果 evidence 只支持一部分，输出 partial。
3. 如果 evidence 不支持或没有提到，输出 not_support。
4. 不要使用外部知识，只能依据 evidence 判断。
5. 输出严格 JSON，不要输出额外文本。

### claim
{claim}

### evidence
{evidence}

### 输出格式
{{
  "support_status": "support | partial | not_support",
  "reason": "简短理由"
}}"""


# AnswerAgent：引用校验失败后的保守修正
ANSWER_CORRECTION_PROMPT = """### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统。此前生成了一份答案，但经过引用校验发现部分内容不被证据支持。

### 原答案
{original_answer}

### 校验发现的问题
{verification_summary}

### 要求
1. 删除或改写所有未被证据支持的句子
2. 只保留引用证据能够明确支持的内容
3. 每个保留的事实句后保留或补充引用编号，格式：事实内容【编号】
4. 如果删除后没有可保留的内容，输出"无答案"
5. 不要编造任何信息

### 问题
{query}

### 修正后的回答
"""
