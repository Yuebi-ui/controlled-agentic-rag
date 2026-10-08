# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------


import os
import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEM3ReRanker(object):
    def __init__(self, model_path, max_length=4096):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()
        self.model.to(self.device)
        self.max_length = max_length
        self._cpu_fallback = False

    def _to_cpu(self):
        self.device = "cpu"
        self.model.to("cpu")
        self._cpu_fallback = True

    def rank(self, query, candidate_docs, topk=10, return_scores=False):
        try:
            return self._rank(query, candidate_docs, topk, return_scores)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and not self._cpu_fallback:
                self._to_cpu()
                return self._rank(query, candidate_docs, topk, return_scores)
            raise

    def _rank(self, query, candidate_docs, topk=10, return_scores=False):
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    scores = self.model(**inputs).logits
            else:
                scores = self.model(**inputs).logits
        scores = scores.detach().cpu().numpy()
        sorted_pairs = sorted(
            zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
        )
        response = [doc for _, doc in sorted_pairs][:topk]
        if return_scores:
            response_scores = [float(score) for score, _ in sorted_pairs][:topk]
            return response, response_scores
        return response


if __name__ == "__main__":
    bge_reranker_large = "./models/BAAI/bge-reranker-v2-m3/"
    # bce_reranker_base = "../../models/bce-reranker-base-v1"
    bge_rerank = BGEM3ReRanker(bge_reranker_large)
    query = "今天天气怎么样"
    docs = ["你好", "今天天气不错", "今天有雨吗"]
    docs = [Document(page_content=doc, metadata={}) for doc in docs]
    response = bge_rerank.rank(query, docs)
    print(response)
