# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

from src.client.llm_client import llm_client, LLM_MODEL_NAME


LLM_HYDE_PROMPT = """
你是一位Tesla汽车专家，现在请你结合Model 3车辆和新能源电动汽车相关知识回答下列问题.
请给出用户问题的使用方法，详细分析问题原因，返回有用的内容。
{query}
最终的回答请尽可能的精简, 不超过100字:
"""


def request_hyde(query):

    prompt = LLM_HYDE_PROMPT.format(query=query) 

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt}
        ],
        top_p=0,
        temperature=0.001
    )
    result = completion.choices[0].message.content

    return result



if __name__ == "__main__":
    query = "介绍一下离车后自动上锁功能"
    res = request_chat(query)
    print(res)
