# Retrieval and parent-aware reranking

| 指标 | Baseline | Hybrid / tuned | 有效样本 |
|---|---:|---:|---:|
|  Child Recall@10 | 0.871 | 0.917 | 240 queries |
|  Parent Recall@10 | 0.863 | 0.896 | 240 queries |
|  Image Recall@10 | 0.600 | 0.800 | 80 labeled |
|  Reranker NDCG@5 | 0.807 | 0.867 | 240 paired pools |

同一个 Query 的 parent/child/image 标签必须按索引粒度分别记录；Image 没有 Gold 的查询不应计为召回失败。固定相同候选集合、top-K、索引版本和评估集才能比较排序模型。用 `retrieval` 与 `reranker` suite 重新计算。
