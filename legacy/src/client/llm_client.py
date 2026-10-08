# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

from openai import OpenAI
from src.config import DASHSCOPE_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME

llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=LLM_BASE_URL
)
