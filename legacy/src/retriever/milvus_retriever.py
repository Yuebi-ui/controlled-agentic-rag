# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------


import time
import logging

logger = logging.getLogger(__name__)
import hashlib
import torch
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker
)
from langchain_core.documents import Document
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction

from src.fields.manual_images import ManualImages
from src.constant import test_doc_path, bge_m3_model_path, milvus_db_path
from src.client.mongodb_config import MongoConfig


EMB_BATCH = 50
#每 50 条文档插入 Milvus 一批
MAX_TEXT_LENGTH = 512 
ID_MAX_LENGTH = 100
COL_NAME = "hybrid_bge_m3" 

mongo_collection = None
_milvus_connected = False
embedding_handler = None

def _lazy_init():
    global mongo_collection, _milvus_connected, embedding_handler
    if mongo_collection is None:
        mongo_collection = MongoConfig.get_collection("manual_text")
    if not _milvus_connected:
        connections.connect(uri=milvus_db_path)
        _milvus_connected = True
    if embedding_handler is None:
        embedding_handler = BGEM3EmbeddingFunction(
            model_name=bge_m3_model_path,
            device="cpu"  # vLLM 占用大量GPU显存，嵌入改用CPU
        )


class MilvusRetriever:
    def __init__ (self, docs, retrieve=False):
        _lazy_init()
        fields = [
            # 构建查询ID，primary key
            FieldSchema(name="unique_id", dtype=DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH),
            # 存储原文，dense vector和sparse vector
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_handler.dim["dense"]),
        ]
        schema = CollectionSchema(fields)

        #删除旧的collection
        if not retrieve and utility.has_collection(COL_NAME):
            Collection(COL_NAME).drop()
        self.col = Collection(COL_NAME, schema, consistency_level="Strong")
        #构建索引，IP是内积
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.col.create_index("sparse_vector", sparse_index)
        self.col.create_index("dense_vector", dense_index)
        self.col.load()

        # 如果是非检索阶段，先构建索引
        if not retrieve:
            self.save_vectorstore(docs)

    #构建索引数据
    def save_vectorstore(self, docs: list[str]): 

        raw_texts = [doc.page_content for doc in docs]
        unique_ids = [doc.metadata["unique_id"] for doc in docs]

        # 计算embedding，会得到稀疏和稠密向量
        texts_embeddings = embedding_handler(raw_texts)

        # batch（分批） embedding 插入
        for i in range(0, len(docs), EMB_BATCH):
            batched_entities = [
                unique_ids[i : i + EMB_BATCH],
                raw_texts[i : i + EMB_BATCH],
                texts_embeddings["sparse"][i : i + EMB_BATCH],
                texts_embeddings["dense"][i : i + EMB_BATCH],
            ]
            self.col.insert(batched_entities)
        logger.info("索引构建完成，插入了 %d 条数据", self.col.num_entities)

    #dense检索
    def dense_search(self, query_dense_embedding, limit):
        search_params = {"metric_type": "IP", "params": {}}
        res = self.col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["unique_id", "text"],
            param=search_params,
        )
        return res


    def sparse_search(self, query_sparse_embedding, limit):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = self.col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["unique_id", "text"],
            param=search_params,
        )
        return res

    #dense检索结果+sparse检索结果+融合排序=最终检索结果
    def hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10,
    ):
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        # 固定权重
        # rerank = WeightedRanker(sparse_weight, dense_weight)
        rerank = RRFRanker()
        res = self.col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["unique_id", "text"]
        )
        return res


    def retrieve_topk(self, query, topk=10):
        t1 = time.time()
        # 抽取query的embedding 
        query_embeddings = embedding_handler.encode_queries([query])

        # 检索Topk
        hybrid_results = self.hybrid_search(
            query_embeddings["dense"][0],
            query_embeddings["sparse"][[0]],
            sparse_weight=0.7,
            dense_weight=1.0,
            limit=topk
        )[0]

        # 关联mongo数据（批量查询避免N+1）
        ids = [result["id"] for result in hybrid_results]
        mongo_results = list(mongo_collection.find({"unique_id": {"$in": ids}}))
        id_to_doc = {r["unique_id"]: r for r in mongo_results}
        related_docs = []
        for result in hybrid_results:
            search_res = id_to_doc.get(result["id"])
            if search_res is None:
                continue
            doc = Document(page_content=search_res["page_content"], metadata=search_res["metadata"])
            related_docs.append(doc)

        return related_docs 


if __name__ == "__main__":
    texts = [k for k in open(test_doc_path).readlines()]
    docs = []
    for text in texts:
        unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        metadata = {"unique_id": unique_id}
        docs.append(Document(page_content=text, metadata=metadata))
    retriever = MilvusRetriever(docs)
    query = "Model3支持的钥匙类型"
    results = retriever.retrieve_topk(query, 2)
    for res in results:
        print(res)
        print("="*100)


