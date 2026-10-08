# -*- coding: utf-8 -*-
# 本地 Citation Judge 客户端 — 指向 Qwen2.5-14B vLLM 服务（端口 :8001）

from openai import OpenAI

JUDGE_BASE_URL = "http://localhost:8001/v1"
JUDGE_API_KEY = "EMPTY"
JUDGE_MODEL_NAME = "Qwen2.5-14B-Instruct-AWQ"

judge_client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
)
