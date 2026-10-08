# -*- coding: utf-8 -*-
# 统一配置管理：从环境变量读取，提供默认值

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# ---- LLM / DashScope ----
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", os.getenv("LLM_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-plus")

# ---- vLLM ----
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", str(BASE_DIR / "LLaMA-Factory-main/output/qwen3_orpo_v4_bf16"))

# ---- MongoDB ----
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "mydatabase")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "manual_text")

# ---- 语义分块服务 ----
SEMANTIC_CHUNK_URL = os.getenv("SEMANTIC_CHUNK_URL", "http://0.0.0.0:6000/v1/semantic-chunks")

# ---- HuggingFace ----
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# ---- 多模态 VL Caption ----
VL_CAPTION_MODEL = os.getenv("VL_CAPTION_MODEL", "qwen-vl-plus")  # DashScope VL 模型，用于图片 Caption
# 注意：Caption 生成是一次性建索引时调用，不影响线上查询
