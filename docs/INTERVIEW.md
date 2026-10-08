# Interview talking points — accurate edition

## 30-second architecture overview

> 这个项目从传统 Agentic Hybrid RAG 升级为受控的 Controlled Agentic RAG。检索先由 BM25、BGE-M3 Dense 和 Chinese-CLIP 三路召回，再做 RRF、Parent Collapse 和 Parent-aware Rerank。真正的控制点不是“Top-K 看起来相关”，而是 Atomic Requirements × Evidence 的充分性判断：三态 Router 决定直接生成、按缺失要求补检、或整体改写。最多两轮，Round2 后对原问题重新精排和门控，充分才封装 final_evidence；最后结构化生成并用 Pydantic 和 Citation Guard 做确定性约束。

## What can be shown from this GitHub branch

- Point directly to `orchestrator.py` to demonstrate the two-round limit and final gate.
- Show `routing.py` for the six-feature contract, loaded *trained* LR interface and conservative safety override.
- Show `retrieval.py` for cross-branch RRF, Parent Collapse and batched canonical evidence expansion.
- Show `citations.py` to explain why a plausible PDF citation from the LLM is never automatically trusted.
- Show synthetic test cases covering enough evidence, partial rescue, failed/rejected, mismatched version, permission and fabricated citation.
- Explain the original models and `data/` were kept privately, while GitHub carries testable orchestration code and reproducible setup.

## Important boundaries to say explicitly

- “当前 GitHub 版本已经实现和离线测试了**核心控制流程**；真实模型、索引和全链路评测仍在逐步接入。”
- “原项目的 SFT 和 Reranker 权重保存着；DPO checkpoint 丢失。现在不宣称 DPO 已部署。”
- “我有历史实验资料中的指标，但当前仓库没有提供完整复现所需的 artifacts，因此把历史结果和本次测试结果分开讲。”
- “Pydantic 和 Citation Guard 能确定性检查结构、身份、版本与权限；**语义上是否真的支持 claim**需要单独评测。”
- “PDF 解析/表格恢复有旧代码和新增设计，但表格的跨页恢复尚不能说已经在真实语料上实测。”

## Avoid overclaiming

Do not say the public repository currently provides real Tesla answering, GPU C4 performance, validated DPO uplift, production-grade authentication, fully automated index lifecycle, or calibrated router probabilities. Those steps require the held-back data/models, integration and benchmarks.

## How to present Evaluation without overclaiming

In GitHub you can show `src/controlled_rag/evaluation/` for metric implementations and `eval/run_all.py` for the no-model reproducibility smoke checks. Explain the difference between **Citation Validity** (allowed ID / canonical page metadata; deterministic) and **Citation Semantic Support** (independent claim-evidence review, cannot be inferred from JSON). The Router metric to emphasize is **False-Sufficient**, not just Accuracy; Round2 is tracked by **trigger**, **eligible rescue denominator**, and **final gate violations**.

If asked for actual DPO-versus-SFT uplift: clearly state the target DPO interface and offline adapter tests exist but DPO weights are missing; the example Base/SFT/DPO evaluation rows are synthetic test fixtures. Give the historical metrics only as historical reported results, **not** verified results of this code.
