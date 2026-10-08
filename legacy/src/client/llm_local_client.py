# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

from openai import OpenAI
from src.config import VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL_NAME as _vllm_model
from src.agents.prompts import ANSWER_GENERATION_PROMPT


llm_client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL
)


def request_chat(query, context, stream=False):

    prompt = ANSWER_GENERATION_PROMPT.format(context=context, query=query)

    completion = llm_client.chat.completions.create(
        model=_vllm_model,
        messages=[
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        frequency_penalty=2.0,
        temperature=0.001,
        top_p=0.95,
        stream=stream,
        timeout=120,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    if not stream:
        result = completion.choices[0].message.content
    else:
        result = completion

    return result



if __name__ == "__main__":

    context = """
    【1】### 离车后自动上锁
    带着手机钥匙或配对的遥控钥匙离开时，车门和行李箱可以自动锁定（如果订购日期是在大约 2019 年 10 月 1 日之后）。要打开或关闭此功能，可点击控制 > 车锁 > 离车后自动上锁。
    **注**：如果已将 Apple 手表认证为钥匙，也可以将该手表用于离车后自动上锁功能。
    【2】车门锁闭时，外部车灯闪烁一次，后视镜折叠（如果折叠后视镜开启）。要在 Model 3 锁定时听到提示音，可点击控制 > 车锁 > 锁定提示音。
    【3】### 大灯延时照明
    停止驾驶并将 Model 3 停在照明较差的环境中时，外部车灯会短暂亮起。它们会在一分钟后或您锁闭 Model 3 时（以较早者为准）自动关闭。当您使用 Tesla 手机应用程序锁定 Model 3 时，大灯将立即熄灭。但是，如果车辆因启用了“离车后自动上锁”功能而锁定（请参阅离车后自动上锁 页码 7），则大灯将在一分钟后自动熄灭。要打开或关闭此功能，请点击控制 > 车灯 > 大灯延时照明。关闭大灯延时照明后，当换入驻车挡并打开车门时，大灯会立即熄灭。"""

    query = "介绍一下离车后自动上锁功能"

    res = request_chat(query, context, stream=True)
    for r in res:
        print(r.choices[0].delta.content, end='')
    print()
