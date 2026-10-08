# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 构建索引：文本 (BM25 + Milvus) + 图片 (Visual Milvus)
# --------------------------------------------


import os
import pickle
from src.parser.pdf_parse import load_pdf, texts_split
from src.parser.image_caption import generate_all_captions
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.retriever.visual_retriever import VisualRetriever
from src.constant import (raw_docs_path, clean_docs_path, split_docs_path,
                          image_save_dir)
from src.client.llm_clean_client import request_llm_clean

# ========================================
# 1. 解析 PDF
# ========================================
raw_docs = load_pdf()
print("文档 page 数:", len(raw_docs))
with open(raw_docs_path, "wb") as f:
    pickle.dump(raw_docs, f)

# ========================================
# 2. 文本清洗和整理
# ========================================
clean_docs = request_llm_clean(raw_docs)
print("清洗后文档 page 数:", len(clean_docs))
with open(clean_docs_path, "wb") as f:
    pickle.dump(clean_docs, f)

# ========================================
# 3. 文档切分
# ========================================
split_docs = texts_split(clean_docs)
print("切分后文档总数:", len(split_docs))
with open(split_docs_path, "wb") as f:
    pickle.dump(split_docs, f)

# ========================================
# 4. 文本索引入库 (BM25 + Milvus)
# ========================================
print("\n--- 文本索引 ---")
bm25_retriever = BM25(split_docs)
candidate_docs = bm25_retriever.retrieve_topk("介绍一下离车后自动上锁功能", topk=3)
print("BM25 召回样例:", [d.page_content[:50] for d in candidate_docs])

milvus_retriever = MilvusRetriever(split_docs)
candidate_docs = milvus_retriever.retrieve_topk("介绍一下离车后自动上锁功能", topk=3)
print("Milvus 召回样例:", [d.page_content[:50] for d in candidate_docs])

# ========================================
# 5. Image Caption 生成（调用 VL API，一次性）
# ========================================
print("\n--- Image Caption ---")
captions = generate_all_captions(image_save_dir, force=False)
print(f"Caption 完成: {len(captions)} 张")

# ========================================
# 6. 图片向量索引入库 (jina-clip-v2 → Milvus)
# ========================================
print("\n--- 图片索引 ---")
image_files = sorted([
    os.path.join(image_save_dir, f)
    for f in os.listdir(image_save_dir)
    if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
])
print(f"图片文件: {len(image_files)} 张")

if image_files:
    visual_retriever = VisualRetriever(image_files)
    # 测试检索
    test_results = visual_retriever.retrieve_topk("充电口", topk=3)
    print("图片检索样例:")
    for doc in test_results:
        print(f"  {doc.metadata.get('image_path', '?')} "
              f"(score={doc.metadata.get('visual_score', 0)})")

print("\n✓ 全部索引构建完成")
