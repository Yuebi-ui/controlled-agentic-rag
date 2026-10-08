# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

import random
import json
import requests
import logging

logger = logging.getLogger(__name__)
import os
import pickle
from typing import List

from src.constant import clean_docs_path


from src.config import SEMANTIC_CHUNK_URL as URL


def request_semantic_chunk(sentences, group_size):
    headers = {
        "Content-Type":"application/json"
    }
    payload = json.dumps({
        "sentences": sentences,
        "group_size": group_size
    })
    try:
        response = requests.post(
            URL,
            headers=headers,
            data=payload,
            timeout=30
        )
        res = response.json()
        text = res["chunks"]
    except Exception as e:
        logger.warning("call rejected: %s", e)
        text = [sentences]
    return text


if __name__ == '__main__':
    data = pickle.load(open("data/processed_docs/clean_docs.pkl", "rb"))
    index = random.sample(range(len(data)), 10)
    for idx in index:
        doc = data[idx].page_content
        res = request_semantic_chunk(doc, 10)
        print("="*100)
        for r in res:
            print(r)
            print("="*100)

