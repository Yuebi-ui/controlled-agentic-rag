# -*- coding: utf-8 -*-
"""
Image Caption 生成器

使用 DashScope qwen-vl-plus API 为图片生成中文描述。
支持 JSON 缓存断点续传，避免重复调用。
"""
import os, json, time, base64
from typing import Optional
from openai import OpenAI

from src import constant
from src.config import DASHSCOPE_API_KEY, LLM_BASE_URL


# 默认 VL 模型
VL_MODEL_NAME = os.getenv("VL_MODEL_NAME", "qwen-vl-plus")

# 缓存文件路径
CAPTION_CACHE_PATH = os.path.join(constant.base_dir, "data/image_captions.json")


def _encode_image(image_path: str) -> str:
    """将本地图片编码为 base64 data URL"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    # 根据扩展名推断 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif"}
    mime = mime_map.get(ext, "jpeg")
    return f"data:image/{mime};base64,{image_data}"


def _load_cache() -> dict[str, str]:
    """加载已有 Caption 缓存"""
    if os.path.exists(CAPTION_CACHE_PATH):
        with open(CAPTION_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, str]):
    """保存 Caption 缓存"""
    os.makedirs(os.path.dirname(CAPTION_CACHE_PATH), exist_ok=True)
    with open(CAPTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _init_client() -> OpenAI:
    """初始化 DashScope 兼容的 OpenAI 客户端"""
    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
    )


CAPTION_PROMPT = """请用简洁的中文描述这张图片的内容。

### 要求
1. 这是特斯拉 Model 3 用户手册中的插图
2. 描述图片中可见的 UI 元素、按钮位置、警告标志、操作示意等
3. 如果有文字标注，请一并描述
4. 限 100 字以内
5. 使用客观陈述语气"""


def generate_caption(image_path: str, client: Optional[OpenAI] = None,
                     model: str = VL_MODEL_NAME) -> str:
    """为单张图片生成中文 Caption"""
    if client is None:
        client = _init_client()

    base64_url = _encode_image(image_path)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_url}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
            max_tokens=200,
            temperature=0.1,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[Caption 生成失败: {e}]"


def generate_all_captions(image_dir: Optional[str] = None,
                          force: bool = False) -> dict[str, str]:
    """
    为图片目录下所有图片生成 Caption，支持缓存断点续传

    Args:
        image_dir: 图片目录，默认使用 data/saved_images/
        force: 是否强制重新生成（忽略缓存）

    Returns:
        {image_filename: caption_text} 字典
    """
    if image_dir is None:
        image_dir = constant.image_save_dir

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    cache = {} if force else _load_cache()
    client = _init_client()

    # 收集所有图片文件
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".gif")
    ])

    total = len(image_files)
    new_count = 0
    skip_count = 0
    fail_count = 0

    print(f"Image Caption 生成: 共 {total} 张图片")
    print(f"模型: {VL_MODEL_NAME}")
    print(f"缓存: {len(cache)} 条已有记录")

    for i, filename in enumerate(image_files):
        # 使用 filename 作为缓存 key
        if filename in cache and cache[filename] and not cache[filename].startswith("[Caption"):
            skip_count += 1
            continue

        image_path = os.path.join(image_dir, filename)
        print(f"  [{i+1}/{total}] {filename} ...", end=" ", flush=True)

        caption = generate_caption(image_path, client=client)
        if caption.startswith("[Caption 生成失败"):
            fail_count += 1
            print(f"❌ {caption}")
        else:
            new_count += 1
            cache[filename] = caption
            print(f"✓ ({len(caption)} 字)")

        # 每张图都即时写入缓存，避免中断丢失
        if new_count > 0 and new_count % 5 == 0:
            _save_cache(cache)

        # API 限速（免费版 1 QPS 足够）
        time.sleep(0.5)

    # 最终保存
    _save_cache(cache)

    print(f"\n完成: 新建 {new_count}, 缓存命中 {skip_count}, 失败 {fail_count}")
    return cache


def get_caption(image_filename: str) -> str:
    """读取单张图片的 Caption（从缓存）"""
    cache = _load_cache()
    return cache.get(image_filename, "")
