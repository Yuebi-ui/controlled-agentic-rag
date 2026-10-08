# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

import os
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from more_itertools import divide
from langchain_core.documents import Document
from src.client.llm_client import llm_client, LLM_MODEL_NAME


MAX_WORKERS = 20

LLM_CLEAN_PROMPT = """
你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。

请根据以下文档内容进行整理：
{}
整理后的输出：
"""


def chat(doc):
    completion = llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "user", "content": doc}
        ],
        top_p=0,
        temperature=0.001,
        timeout=120
    )
    return completion.choices[0].message.content


def request_llm_clean(docs):
    clean_docs = []
    docs_mapping = {doc.metadata['unique_id']: doc for doc in docs}
    docs_groups = [list(group) for group in divide(MAX_WORKERS, docs)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for groups in docs_groups:
            futures = {doc.metadata['unique_id']: executor.submit(chat,
                LLM_CLEAN_PROMPT.format(doc.page_content)) for doc in groups}

            for unique_id in tqdm(futures):
                future = futures[unique_id]
                result = future.result()
                if result is None:
                    continue
                clean_docs.append(
                   Document(page_content=result, metadata=docs_mapping[unique_id].metadata)
                )
    return clean_docs


if __name__ == "__main__":
    doc = "".join(open("./data/ut/test_docs.txt").readlines())
    res = chat(LLM_CLEAN_PROMPT.format(doc))
    print(res)
