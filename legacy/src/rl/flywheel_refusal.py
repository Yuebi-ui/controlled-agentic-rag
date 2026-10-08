# -*- coding: utf-8 -*-
"""RL 飞轮：构造拒答修正偏好对

1. 合成 100 条无关中文问题（应拒答场景）
2. 用当前模型生成答案 → 找出模型回答了的（幻觉）
3. 构造 chosen=拒答 / rejected=模型错误答案
4. 合并到现有偏好对数据集
"""
import json, sys, os, re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from openai import OpenAI

VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "orpo-v3")
PREF_PATH = PROJECT_DIR / "LLaMA-Factory-main/data/rag_preference.json"
OUTPUT_PATH = PROJECT_DIR / "LLaMA-Factory-main/data/rag_preference_v4.json"

SYSTEM_PROMPT = "你是特斯拉电动汽车Model 3车型的用户手册问答系统，请严格基于提供的信息回答用户问题。"

# ── 合成无关问题 ──
SYNTHETIC_QUESTIONS = [
    # 日常生活
    "怎么做红烧肉？", "感冒了应该吃什么药？", "最近有什么好看的电影推荐吗？",
    "今天天气怎么样？", "哪个牌子的手机好？", "怎么做蛋糕？",
    "推荐几个旅游景点", "怎么减肥最有效？", "怎么学好英语？",
    "股票现在买什么好？", "哪家外卖好吃？", "怎么选笔记本电脑？",
    # 娱乐八卦
    "你觉得哪个明星最漂亮？", "世界杯冠军是谁？", "LOL怎么上分？",
    "最近有什么好听的歌？", "漫威电影观看顺序", "王者荣耀什么英雄强？",
    # 通用问题
    "人生的意义是什么？", "帮我写一封辞职信", "翻译：hello world",
    "给我讲个笑话", "你会取代人类吗？", "你有感情吗？",
    "帮我写一首诗", "1+1等于几？", "你是谁？",
    "推荐一本书", "今天几号？", "帮我做数学题",
    # 其他领域
    "房贷利率现在是多少？", "怎么办理护照？", "驾照怎么考？",
    "什么是区块链？", "怎么炒股票？", "比特币还能涨吗？",
    "怎么养多肉植物？", "猫粮什么牌子好？", "怎么给宝宝取名？",
    "怎么修复感情裂痕？", "失眠怎么办？", "腰疼怎么缓解？",
    # 特斯拉但超出范围
    "特斯拉股票现在多少钱？", "马斯克最近在干什么？",
    "Model 3和蔚来ET5哪个好？", "特斯拉工厂在哪里？",
    "买特斯拉股票能赚钱吗？", "马斯克有几个孩子？",
    # 无意义/闲聊
    "你好", "在吗", "哈哈哈", "谢谢", "再见",
    "今天心情不好，陪我聊聊", "你觉得我帅吗？", "说点好听的",
    "你是不是傻？", "给我发红包", "帮我查个人",
    "我想听他的歌曲", "我要去那", "还珠格格的",
    # 更多领域外问题
    "怎么注册公司？", "Python和Java哪个好？", "怎么学钢琴？",
    "哪家健身房好？", "怎么去除甲醛？", "机票怎么订便宜？",
    "怎么给宝宝断奶？", "什么防晒霜好用？", "怎么挑选钻石？",
    "大学选什么专业好？", "怎么和领导谈加薪？", "怎么追女生？",
    "量子力学是什么？", "黑洞里面有什么？", "外星人存在吗？",
    "怎么种菜？", "什么鱼好养？", "怎么训练狗狗？",
    "怎么去除冰箱异味？", "洗衣机怎么选？", "空调什么牌子好？",
    "什么电动牙刷好用？", "蓝牙耳机推荐", "机械键盘推荐",
    "怎么选保险？", "信用卡哪家好？", "怎么理财？",
    "怎么选SUV？", "什么行车记录仪好？", "如何投诉4S店？",
    "帮我写个借条", "劳动合同怎么看？", "怎么退个人所得税？",
]

REFUSAL_ANSWERS = [
    "抱歉，我是特斯拉Model 3用户手册问答助手，无法回答这个问题。请询问与Model 3车辆使用相关的问题。",
    "根据提供的信息，无法确定该问题的答案。",
]


def main():
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)

    # ── Step 1: 用模型跑所有合成问题 ──
    print(f"Step 1: 用 {MODEL_NAME} 跑 {len(SYNTHETIC_QUESTIONS)} 条合成问题...")
    hallucination_pairs = []
    refused_count = 0
    answered_count = 0

    for i, q in enumerate(SYNTHETIC_QUESTIONS):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=256,
            temperature=0.1,
            top_p=0.95,
            extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = resp.choices[0].message.content.strip()

        # 判断是否拒答
        refusal_patterns = ["无答案", "无法回答", "抱歉", "无法提供", "无法确定",
                            "只能回答", "请询问", "无法执行"]
        is_refusal = any(p in answer for p in refusal_patterns)

        if is_refusal:
            refused_count += 1
        else:
            answered_count += 1
            # 构造修正对：chosen=拒答, rejected=模型的错误答案
            pair = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": q},
                ],
                "chosen": {"from": "gpt", "value": REFUSAL_ANSWERS[0]},
                "rejected": {"from": "gpt", "value": answer},
            }
            hallucination_pairs.append(pair)
            if len(hallucination_pairs) <= 5:
                print(f"  幻觉[{i}]: Q={q[:30]}... A={answer[:60]}...")

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(SYNTHETIC_QUESTIONS)}] 拒答={refused_count} 回答={answered_count}")

    print(f"\nStep 1 完成: 拒答={refused_count} 幻觉={answered_count}")

    # ── Step 2: 合并现有偏好对 ──
    print(f"\nStep 2: 合并现有偏好对...")
    with open(PREF_PATH) as f:
        existing_pairs = json.load(f)
    print(f"  现有偏好对: {len(existing_pairs)}")
    print(f"  新增拒答对: {len(hallucination_pairs)}")

    all_pairs = existing_pairs + hallucination_pairs
    print(f"  合并后: {len(all_pairs)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n保存到: {OUTPUT_PATH}")

    # ── Step 3: 更新 dataset_info.json ──
    info_path = PROJECT_DIR / "LLaMA-Factory-main/data/dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["rag_preference_v4"] = {
        "file_name": "rag_preference_v4.json",
        "ranking": True,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"注册 rag_preference_v4 到 dataset_info.json")


if __name__ == "__main__":
    main()
