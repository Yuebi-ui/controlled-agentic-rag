# Controlled Round2 & Final Evidence Gate



- 总共 `240` 个 Query，触发二轮 `140` 个。
- Rescue `79.7%`；分母为 Round2 触发、Gold 可回答且 Round1 未充分的案例。

协议：原子需求分解 → R1 Gold coverage 与模型路由预测分别保存 → 部分缺证据则重写缺失需求，失败则重写 Query → R2 后 merge/dedup 并保留 R1 的强相关证据 → 原问题 rerank + final gate → 允许生成或拒答。报告 Trigger Precision/Recall、Rescue 分母、Coverage gain(pp)、二轮新增时延与不能恢复的 Bad Case。不得把 Answerability Decision F1 当成自然语言答案 Correctness。
