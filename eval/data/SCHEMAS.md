# Evaluation JSONL Contract v1

每行一个严格 JSON object，`id` 为用例唯一标识（Generation 允许同一 ID 对应不同模型，但 `(model_variant,id)` 必须唯一）。不允许未知字段、无穷大、NaN、伪造标签。真正的数据集应另附其版本、来源、许可、Annotator Protocol 及 Split Audit 记录。

## 最小字段速览

| Suite | 必需字段 | 可选字段/说明 |
|---|---|---|
| `retrieval` | `id`, `gold_parent_ids`, `ranked_parent_ids`（有图像 Gold 时可只有 Image） | `gold_image_ids`, `ranked_image_ids`；Gold 必须至少有一种非空 |
| `reranker` | `id`, `positive_ids`, `before`, `after` | `after_scores`，必须同候选 ID 集合；pairwise 可以不计算 |
| `router` | `id`, `gold_state`, `predicted_state` | `raw_model_state`, `probabilities`（3 类和为 1） |
| `round2` | `id`, `round1_state`, `retrieval_rounds`, `final_status`, `gold_answerable`, `requirement_count`, `round1_covered`, `final_covered` | `round2_state` 两轮时必填；`mandatory_round1_keys`, `merged_evidence_keys`, `final_evidence_keys` |
| `generation` | `id`, `model_variant`, `gold_answerable`, `schema_valid` | `predicted_abstain`（合规则必填），`correctness`, `faithfulness` 只能基于答案进行独立评审 |
| `citation` | `id`, `allowed_evidence_ids`, `canonical_evidence`, `citations` | 引用包含 `evidence_id/document_id/source_uri/source_pages/image_uri?`; `semantic_supported` 缺省为 null；`gold_required_evidence_ids` |
| `safety` | `id`, `attack_type`, `expected_status`, `actual_status`, `actual_reason_code` | `expected_reason_codes`, `must_not_generate`, `generation_called`, `leaked_unauthorized`, `attack_succeeded`；Injection 需标注成功与否 |
| `benchmark` | `id`, `scenario`, `start_ms`, `end_ms`, `http_status`, `concurrency` | `gpu_memory_mb` 真实测量才填 |
| `workflow` | `id`, `gold_answerable`, `gold_round1_state`, `expected_round2`, `actual_round2`, `retrieval_rounds`, `final_status`, `reason_code`, `sealed_final_present`, `elapsed_ms` | `predicted_round1_state`, `final_gate_state`, `mandatory_round1_keys`, `merged_evidence_keys`；由实际 Pipeline Trace 生成 |
| `ablation` | `id`, `variant`, `metrics` | 必须存在 `baseline` 和对照；每个指标以其有标注的 ID 子集配对比较 |
| `badcase` | `id`, `category`, `severity`, `expected_status`, `actual_status`, `actual_reason_code`, `actual_retrieval_rounds` | `expected_reason_codes`, `forbidden_citation_ids`, `observed_citation_ids`, `max_allowed_rounds` |
| `split-audit` | `id`, `split`, `document_id` | `text_hash` 用于 exact content overlap 检测 |

`eval/agent_workflow/run.py` 接受的**输入 Gold Case**（不同于 `workflow` 输出记录）需要 `id`, `query`, `gold_answerable`, `gold_round1_state`, `expected_round2`；可选 `gold_parent_ids` 和 `gold_image_ids`。两个 Gold 路由/触发字段应由对相应第一轮证据的独立标注给出，不从 Pipeline prediction 推断。`eval/ablation/run_matrix.py` 的 `matrix.json` 必须包含 `baseline` 和至少两个不同配置的 `variants: [{name, settings}]`。

## 核心分母与缺失政策

- Parent Recall@K：每题 `命中 Gold Parent 数 / Gold Parent 数`，再按具有 Parent 标注的 Query 等权平均。MRR@K 是第一个相关 Parent 的倒数排名，无命中记 0。
- Image Recall@K：只计算有图像 Gold 的题；**没有图像标注输出 null，而非 0**。
- NDCG@K：二元相关性，`DCG / Ideal DCG`，IDCG 以每道题实际 Gold 数与 K 的较小值计算。
- False-Sufficient：`Gold 属于 Partial/Failed 且 Predicted= Sufficient 的数目 / Gold 属于 Partial/Failed 的总数`。它不等于普通 Accuracy 的补数。没有不足样本则 `null`。
- 多类 Brier：`mean(Σ_c (p_c - one_hot_c)^2)`，不额外按 C 缩放。ECE 为模型 Top1 置信度在 10 个等宽 bins 的频数加权校准误差。
- Round2 Trigger：`两轮检索数 / 已至少进入第一轮检索数`；Rescue（Answerable triggered）：`原本不足+触发且 Gold 可回答，最终 Sufficient 并回答数 / 原本不足+触发且 Gold 可回答数`。
- Round2 保留：已被第一轮认定覆盖的 evidence_key 要保存在 Round2 **merge candidate pool**。进入最终答案的 TopK 可能因重排改变，是否覆盖由最终 Gate 再判。
- Correctness/Faithfulness：只有独立审核了回答的样本参与分母。若没有标注，指标为 `null`，并给出 annotation count。
- Citation Validity：只有合法 sealed ID 且文档 ID、源 URI、页码、Image URI（如有）与 canonical metadata 全部一致才合法。Citation Semantic Support 要逐引用独立检查，缺标注时 `null`。
- HTTP 2xx 为成功，非 2xx 纳入总吞吐和非 2xx 率；P50/95/99 使用 nearest-rank，所有 timestamp 必须源自同一个 monotonic clock。
- Agent Workflow Trigger：TP/FP/FN/TN 以独立 `expected_round2` 为 Gold；Precision 分母为实际触发数，Recall 分母为应该触发数，缺少分母时返回 null。
- Agent Workflow Answerability：`answered` 与 Gold answerable 对齐；`failed` 视为不正确决策。指标不是 Correctness/Faithfulness，最终门控违规只检验控制状态。
- Ablation Bootstrap：按每个指标双方均有观测的共同 Query ID 配对，缺失 Gold 不填 0；固定随机种子，CI 只是成对重抽样的描述性区间，不代替严格显著性分析。
- Release Gate 缺失路径、null 或非数值一律失败；`delta = candidate - baseline`。例如 `min:-0.02` 是允许 2 个百分点以内的下降（如果基础指标单位为比例）。

## 警告

🔴 所有 `*_synthetic.jsonl` **完全虚构**，专门测试统计和门禁边界。`generation_synthetic.jsonl` 的 Base/SFT/DPO 标签不对应任何真实模型；DPO Checkpoint 缺失。`benchmark_synthetic.jsonl` 的耗时、显存是人为构造值。发布时不能将这些数据当作用户项目的实测指标。
