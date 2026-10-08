# -*- coding: utf-8 -*-
# QueryAgent 输出结构校验

from pydantic import BaseModel, Field
from typing import Literal


class QueryRewriteResult(BaseModel):
    original_query: str = ""
    rewritten_query: str = ""
    need_retrieval: bool = True
    reasoning: str = ""
    query_type: Literal["fact_qa", "compare", "summary", "multi_hop", "other"] = "fact_qa"
    intent: Literal["knowledge_qa", "chitchat", "action_request", "ambiguous"] = "knowledge_qa"
