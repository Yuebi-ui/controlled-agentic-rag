# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------


import re
import fitz
#PyMuPDF，用于读取 PDF 文本、页面、图片
import json
import copy
import hashlib
import tiktoken
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection
from pymongo import UpdateOne
from typing_extensions import List

from src import constant
from src.fields.manual_images import ManualImages
from src.fields.manual_info_mongo import ManualInfo
from src.client.mongodb_config import MongoConfig
import src.parser.image_handler as image_handler
from src.client.semantic_chunk_client import request_semantic_chunk


# 全局配置
_chunk_size = 256
_chunk_overlap = 50
_min_filter_pages = 4
_max_filter_pages = 247
_semantic_group_size = 10
_max_parent_size = 512
_page_clip = 50
encoding = tiktoken.get_encoding("cl100k_base")
manual_text_collection: Collection = MongoConfig.get_collection("manual_text")
file_path = constant.pdf_path


# ===== TextSplitter 设置 =====

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunk_size,
    chunk_overlap=_chunk_overlap,
    # 按这个优先级递归切
    separators=["\n\n", "\n"],
    length_function=lambda text: len(encoding.encode(text))
)


# ===== 文本预处理部分 =====

def sentence_split(text: str) -> list[str]:
    """按中文/英文标点切句"""
    sentences = re.split(r'(?<=[。\n\t])+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def load_pdf() -> list[Document]:
    pdf = fitz.open(file_path)
    raw_docs = []

    for idx, page_num in enumerate(tqdm(range(len(pdf)))):
        # 过滤封面和目录
        if idx < _min_filter_pages or idx > _max_filter_pages:
            continue

        page = pdf.load_page(page_num)
        crop = fitz.Rect(0, 0, page.rect.width, page.rect.height-_page_clip)
        text = page.get_text(clip=crop)
        images = page.get_images(full=True)

        manual_images_list: List[ManualImages] = []
        for img_index, img in enumerate(images):
            manual_image: ManualImages = image_handler.handle_image(img, img_index, page)
            #取图片、保存图片文件、尝试识别图片附近标题、返回图片结构化信息
            if manual_image:
                manual_images_list.append(json.loads(manual_image.json()))

        if text.strip():
            unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            #如果文本不为空，就对这一页文本生成 MD5。这个 MD5 是 page-level 文档的唯一 ID。
            metadata = {
                "unique_id": unique_id,
                "source": file_path,
                "page": page_num + 1,
                "images_info": manual_images_list
            }

            raw_docs.append(Document(page_content=text, metadata=metadata))

    return raw_docs


def texts_split(raw_docs: list[Document]) -> list[Document]:
    """句子级 + 语义感知切分"""
    all_split_docs = []

    for doc in tqdm(raw_docs):

        # 语义切分
        grouped_chunks = request_semantic_chunk(doc.page_content, group_size=_semantic_group_size)
        #以10个句子/段落为一组，但实际还是要看semantic_chunk_client 和服务端 semantic_chunk.py 的实现。

        # 父doc
        parent_docs = []
        for group in grouped_chunks:
            parent_id = hashlib.md5(group.encode('utf-8')).hexdigest()
            parent_metadata = copy.deepcopy(doc.metadata)
            parent_metadata["unique_id"] = parent_id 
            parent_doc = Document(page_content=group, metadata=parent_metadata)
            parent_docs.append(parent_doc)
            if len(group) < _max_parent_size:
                all_split_docs.append(parent_doc)
        save_2_mongo(parent_docs)

        # 子doc
        for chunk in parent_docs:
            # 带overlap继续句子级切分
            split_docs = text_splitter.create_documents([chunk.page_content], metadatas=[chunk.metadata])
            reid_split_docs = []
            for child_doc in split_docs:
                child_id = hashlib.md5(child_doc.page_content.encode('utf-8')).hexdigest()
                if child_doc.page_content == chunk.page_content:
                    continue
                child_metadata = copy.deepcopy(chunk.metadata)
                child_metadata["unique_id"] = child_id
                child_metadata["parent_id"] = chunk.metadata["unique_id"]
                reid_child_doc = Document(page_content=child_doc.page_content, metadata=child_metadata)
                reid_split_docs.append(reid_child_doc)

            save_2_mongo(reid_split_docs)
            all_split_docs.extend(reid_split_docs)

    return all_split_docs


def save_2_mongo(split_docs):
    if not split_docs:
        return
    operations = []
    for doc in split_docs:
        metadata = doc.metadata

        unique_id = metadata.get("unique_id")
        if not unique_id:
            continue

        doc_record = ManualInfo(
            unique_id=unique_id,
            page_content=doc.page_content,
            metadata=metadata
        )

        operations.append(UpdateOne(
            {"unique_id": doc_record.unique_id},
            {"$set": doc_record.model_dump()},
            upsert=True
        ))

    if operations:
        manual_text_collection.bulk_write(operations, ordered=False)


