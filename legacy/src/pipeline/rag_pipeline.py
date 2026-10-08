# -*- coding: utf-8 -*-
# RAGPipeline — 统一 Agentic RAG 管线
#
# 供 CLI (infer_agentic.py)、Web Demo (web_demo.py)、FastAPI (src/api/main.py)、
# 评测 (eval/) 共用同一套问答逻辑。

import time

from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.retriever.visual_retriever import VisualRetriever
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing

from src.client.llm_client import llm_client as remote_llm, LLM_MODEL_NAME as remote_model
from src.client.llm_local_client import llm_client as local_llm
from src.config import VLLM_MODEL_NAME as qwen3_8b_tune_model_name

from src.agents.query_agent import QueryAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.retrieval_config import get_retrieval_config
from src.agents.evidence_agent import EvidenceAgent
from src.agents.answer_agent import AnswerAgent
from src.agents.citation_verifier import CitationVerifier


class RAGPipeline:
    """统一 Agentic RAG 管线，所有入口共用"""

    def __init__(self):
        self.bm25 = BM25(docs=None, retrieve=True)
        self.milvus = MilvusRetriever(docs=None, retrieve=True)
        # 图片检索器（可选，建索引后可用）
        self.visual = self._init_visual()
        self.reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
        self.query_agent = QueryAgent(llm_client=remote_llm, model_name=remote_model)
        self.retrieval_agent = RetrievalAgent(self.bm25, self.milvus, self.visual)
        self.evidence_agent = EvidenceAgent(llm_client=remote_llm, model_name=remote_model)
        self.answer_agent = AnswerAgent(llm_client=local_llm, model_name=qwen3_8b_tune_model_name)
        self.citation_verifier = CitationVerifier(llm_client=remote_llm, model_name=remote_model)

    def _init_visual(self):
        """初始化图片检索器。如果索引未构建，降级为 None"""
        try:
            return VisualRetriever(retrieve=True)
        except Exception:
            return None

    @staticmethod
    def _default_result(query: str) -> dict:
        """统一返回结构，所有分支基于此更新字段"""
        return {
            "query": query,
            "answer": "",
            "cite_pages": [],
            "related_images": [],
            "rewritten_query": query,
            "query_type": "fact_qa",
            "intent": "knowledge_qa",
            "rewrite_reason": "",
            "need_retrieval": True,
            "bm25_count": 0,
            "milvus_count": 0,
            "merged_count": 0,
            "retrieval_time": "0.00s",
            "ranked_doc_count": 0,
            "evidence_enough": False,
            "evidence_reason": "",
            "self_rag": False,
            "suggested_query": "",
            "second_merged_count": 0,
            "raw_answer": "",
            "citation_verified": True,
            "citation_unsupported": 0,
            "citation_partial": 0,
            "citation_rewrite_triggered": False,
            "citation_details": [],
            "total_time": 0.0,
            "_bm25_docs": [],
            "_milvus_docs": [],
            "_ranked_docs": [],
            "_ranked_scores": [],
        }

    def run(self, query: str, stream: bool = False,
            stream_callback: callable = None) -> dict:
        """执行完整管线。

        Args:
            query: 用户问题
            stream: 是否流式生成答案
            stream_callback: 流式回调 fn(token_text) — 仅在 stream=True 时生效

        Returns:
            dict: 包含 answer、中间结果、耗时等完整信息
        """
        result = self._default_result(query)
        t0 = time.time()

        # Step 1: QueryAgent
        q_info = self.query_agent.run(query)
        result["rewritten_query"] = q_info["rewritten_query"]
        result["query_type"] = q_info["query_type"]
        result["rewrite_reason"] = q_info.get("reasoning", "")
        result["need_retrieval"] = q_info.get("need_retrieval", True)
        result["intent"] = q_info.get("intent", "knowledge_qa")
        search_query = q_info["rewritten_query"]

        # 非知识问答意图：跳过检索管线，直接生成引导性回复
        intent_reply_map = {
            "chitchat": "抱歉，我是Model 3用户手册问答助手，只能回答与Model 3车辆使用相关的问题。请问有什么关于Model 3的使用问题我可以帮您解答？",
            "action_request": "抱歉，我无法执行该操作。我是Model 3用户手册问答助手，只能回答与Model 3车辆使用相关的问题。请问有什么关于Model 3的使用问题我可以帮您解答？",
            "ambiguous": "抱歉，我不太确定您想了解什么。请具体描述您想了解的Model 3功能或使用问题，我会尽力帮您解答。",
        }
        intent = result["intent"]
        if intent in intent_reply_map:
            result["answer"] = intent_reply_map[intent]
            result["total_time"] = round(time.time() - t0, 2)
            return result

        if not q_info.get("need_retrieval", True):
            answer = self.answer_agent.generate(query, [], stream=False)
            result["answer"] = answer
            result["total_time"] = round(time.time() - t0, 2)
            return result

        # Step 2: Retrieval
        config = get_retrieval_config(q_info["query_type"])
        ret = self.retrieval_agent.retrieve(search_query, config["bm25_topk"], config["milvus_topk"])
        result["bm25_count"] = len(ret["bm25_docs"])
        result["milvus_count"] = len(ret["milvus_docs"])
        result["merged_count"] = len(ret["merged_docs"])
        result["retrieval_time"] = f"{ret['elapsed']:.2f}s"
        result["_bm25_docs"] = ret["bm25_docs"]
        result["_milvus_docs"] = ret["milvus_docs"]
        result["_visual_docs"] = ret.get("visual_docs", [])

        # Step 3: Rerank（空召回兜底）
        if not ret["merged_docs"]:
            result["answer"] = "无答案"
            result["evidence_reason"] = "未召回到相关文档"
            result["total_time"] = round(time.time() - t0, 2)
            return result

        ranked_docs, ranked_scores = self.reranker.rank(
            search_query, ret["merged_docs"], config["rerank_topk"], return_scores=True,
        )
        result["ranked_doc_count"] = len(ranked_docs)

        # Step 4: Evidence + Self-RAG
        ev = self.evidence_agent.judge(query, q_info["query_type"], ranked_docs, ranked_scores)
        result["evidence_enough"] = ev["is_enough"]
        result["evidence_reason"] = ev["reason"]
        result["self_rag"] = False

        if not ev["is_enough"]:
            result["self_rag"] = True
            sc = get_retrieval_config(q_info["query_type"], evidence_enough=False)
            sq = ev.get("suggested_query") or search_query
            result["suggested_query"] = sq
            ret2 = self.retrieval_agent.retrieve(sq, sc["bm25_topk"], sc["milvus_topk"])
            all_merged = merge_docs(ret["merged_docs"], ret2["merged_docs"])
            ranked_docs, ranked_scores = self.reranker.rank(
                sq, all_merged, sc["rerank_topk"], return_scores=True,
            )
            result["second_merged_count"] = len(all_merged)

        # Step 5: Answer
        if stream and stream_callback:
            response = ""
            res_handler = self.answer_agent.generate(query, ranked_docs, stream=True)
            for r in res_handler:
                try:
                    token = r.choices[0].delta.content
                except (AttributeError, IndexError, KeyError):
                    continue
                if token is None:
                    continue
                response += token
                stream_callback(token)
        else:
            response = self.answer_agent.generate(query, ranked_docs, stream=False)

        result["raw_answer"] = response

        # Step 6: Citation Verification
        cv = self.citation_verifier.verify(response, ranked_docs, query)
        result["citation_rewrite_triggered"] = False

        if not cv["verified"] and "无答案" not in response:
            result["citation_rewrite_triggered"] = True
            revised = self.answer_agent.rewrite_with_supported_evidence(
                query=query,
                original_answer=response,
                ranked_docs=ranked_docs,
                verification=cv,
            )
            cv = self.citation_verifier.verify(revised, ranked_docs, query)
            response = revised

        result["citation_verified"] = cv["verified"]
        result["citation_unsupported"] = cv["unsupported_count"]
        result["citation_partial"] = cv.get("partial_count", 0)
        result["citation_details"] = cv["claim_results"]

        # Step 7: Post Processing（加入图片检索结果）
        visual_docs = result.get("_visual_docs", [])
        answer_info = post_processing(response, ranked_docs, visual_docs)
        result["answer"] = answer_info["answer"]
        result["cite_pages"] = answer_info["cite_pages"]
        result["related_images"] = answer_info["related_images"]
        result["total_time"] = round(time.time() - t0, 2)

        # 保留精排文档供上游使用
        result["_ranked_docs"] = ranked_docs
        result["_ranked_scores"] = ranked_scores

        return result
