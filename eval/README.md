# Evaluation & Reproducibility / 评测体系

这套评测代码与**旧仓库 `legacy/eval/`** 是两条独立路径。现用的评测计算实现在 `src/controlled_rag/evaluation/`，其输入是有 Schema 的 JSONL；`eval/` 放置可公开的**完全虚构**样本、演示入口和 CI 所需的控制门禁。**示例文件中的 DPO 行不是实际模型的结果**。

## 快速运行（不需要 GPU、真实 PDF 或模型）

在仓库根目录：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python eval/run_all.py
python scripts/collect_eval_traces.py --mode synthetic \
  --cases eval/data/pipeline_cases_synthetic.jsonl --output outputs/eval_trace
python -m controlled_rag.evaluation --suite round2 \
  --input outputs/eval_trace/round2.jsonl --output outputs/round2_metrics.json
python eval/safety/run_control_probes.py --output outputs/guard_probes.jsonl
python -m controlled_rag.evaluation --suite safety \
  --input outputs/guard_probes.jsonl --output outputs/guard_metrics.json
```

`eval/run_all.py` 生成 `eval/reports/synthetic-run/combined.json` 以及 `synthetic_gate.json`。这只证明**样本格式、计算公式和离线控制机制**按预期运行；不要将其中指标用于简历或面试的真实模型性能陈述。生成报告默认不加入 Git。

## 评测模块与指标

| Suite | 脚本输入与主要指标 | 关键约束 |
|---|---|---|
| `retrieval` | Parent/Image Recall、Hit、MRR、NDCG @1/5/10 | Parent/Image 分开，逐 Query 宏平均；无图像标注输出 `null` |
| `reranker` | Before/After NDCG、MRR、Recall、pairwise accuracy | 同候选集合比较；相同分数 pairwise 记 0.5 |
| `router` | 三态混淆矩阵、Macro-F1、False-Sufficient、Brier、ECE | FS 分母是 Gold 非 Sufficient；多类 Brier 不做除以类别数缩放；ECE 为 Top-label 10 bins |
| `round2` | Trigger、Rescue、原子需求 Coverage Gain、最终 Gate 违规 | Rescue 分母和触发率分母显式分开；必须保留 Round1 已覆盖证据于 **merge pool** |
| `generation` | Base/SFT/DPO JSON 合规率、拒答 P/R/F1、正确率、忠实率 | 语义指标必须有**独立标注**；同问题 ID 跨模型对照，报告样本对齐情况 |
| `citation` | ID+来源元数据 Validity、标注语义支持率、Gold Evidence Recall | **合法引用 ≠ 语义支持**；无标注显示 `null` |
| `safety` | 按攻击类型通过率、失败 Case ID | Prompt Injection 需要人工/独立评审 `attack_succeeded`，仅看 HTTP 状态不算防护通过 |
| `benchmark` | P50/P95/P99、非 2xx、QPS、可选显存 | 时间戳必须共用一套单调时钟；没有 GPU 采样不能填写 GPU 结果 |
| `ablation` | 同 Query ID 配对均值差 + Bootstrap CI | 单独输出失配样本数量，不将不等样本集硬比较 |
| `badcase` | 固定 Bad Case 行为回归、Reason Code、违规轮数、禁用引用 | 高风险失败会进入 Release Gate；Case ID 稳定，避免以重新训练来掩盖回归 |
| `split-audit` | 跨 Train/Dev/Test 的 ID、Document、原文哈希泄漏 | 只能检查精确 ID/哈希和文档归属，不能假装检测到全部语义近重复 |
| `release` | 绝对门槛、相对 Baseline 变化 | 任何缺失或非数值指标一律 Gate Fail；Delta = Candidate − Baseline |

指标、Schema 和缺失值政策详见 [`data/SCHEMAS.md`](data/SCHEMAS.md)。

## 单项指标计算

```bash
python -m controlled_rag.evaluation --suite retrieval \
  --input eval/data/retrieval_synthetic.jsonl \
  --output outputs/retrieval_demo.json --source synthetic
```

真正拿到私有数据和模型权重后，改成 `--source real_unverified`。只有已经审阅数据版本、标签来源、同源样本划分和运行产物的工作流，才可自行声明 `--source real_audited`；程序不能自动验证这项声明。报告记录原始输入 SHA256、运行 UTC 时间、可选 model/index version，便于审计复现。

## 实际 Pipeline trace → 评测

`ControlledRAGPipeline.run(query, request, observer=TraceRecorder())` 允许记录**不含原问题/证据正文**的阶段事件，包括 R1/R2 Candidate、Rerank 完整排序、Router 覆盖、Final Gate、sealed IDs、结果状态。系统不会自动代填 Gold Labels：

```bash
python scripts/collect_eval_traces.py --mode synthetic \
  --cases eval/data/pipeline_cases_synthetic.jsonl \
  --output outputs/eval_trace
```

后续接通本地真实检索/模型后，开发者创建自己的 `module:function` factory，返回 `(pipeline, request_factory)`，再运行 `--mode real --factory my_project.eval_factory:build`。这个 factory **目前并未随仓库提供**，否则会错误地暗示 Mongo、Milvus、Router、DPO 都已联通。

只有在 Query 有真实 Gold Parent/Image 标签时，采集器才输出 Recall 用的记录；没有可解释的语义标注时，Citation trace 的 `semantic_supported` 为空。`outputs/` 中可能含有来源 URI，必须留在私有环境并人工审查脱敏。

## 三模型真实生成对比（待模型可用后运行）

```bash
# 用户自己的不可公开 sealed evidence 问题集；需要对应真实模型服务。
python eval/generation/run_live.py --variant sft \
  --input /private/generation_inputs.jsonl --output outputs/sft.jsonl
python eval/generation/run_live.py --variant dpo \
  --input /private/generation_inputs.jsonl --output outputs/dpo.jsonl
python -m controlled_rag.evaluation --suite generation \
  --input outputs/combined_generation_with_independent_annotations.jsonl \
  --output outputs/generation_metrics.json --source real_unverified
```

`run_live.py` 可以显式选择 `base/sft/dpo`；默认 DPO。DPO 权重缺失时程序抛出错误并停止，**不会假装 DPO 测试完成，也不会静默降级到 SFT**。这个入口仅评测生成侧 JSON 和拒答，真实语义评测还需人工标注以及完整 Pipeline 的 Citation Guard 验证。合并不同模型输出时，要按 `(id, model_variant)` 对齐，并排除跨训练集泄漏的样本。

## 安全与容量

```bash
# 不需要模型的确定性防护探针
python eval/safety/run_control_probes.py --output outputs/safety_probes.jsonl

# 真实启动的本地服务压力测试；用的是虚构公开问题
python scripts/http_bench.py --base-url http://127.0.0.1:9000 \
  --requests 30 --concurrency 2 --output outputs/bench_local.jsonl
python -m controlled_rag.evaluation --suite benchmark \
  --input outputs/bench_local.jsonl --output outputs/bench_metrics.json

# 可选需要 locust: pip install -e '.[benchmark]'
locust -f eval/benchmark/locustfile.py --host http://127.0.0.1:9000 \
  --headless -u 2 -r 1 -t 30s --csv outputs/locust/synthetic
```

Locust 默认统计保存在自己的 CSV 格式；此处独立 JSONL 计算器使用 `http_bench.py` 的时钟一致原始记录。**不能直接将 Locust 聚合 CSV 当成 request-level JSONL**。请求失败同样入样本，吞吐不会只计算成功请求。

## 回归门禁

```bash
python -m controlled_rag.evaluation --suite release \
  --input eval/reports/synthetic-run/combined.json \
  --rules eval/regression/synthetic_control_gate.json \
  --output outputs/synthetic_release_check.json
```

`eval/regression/release_gate.example.json` 展示未来真实版本的阈值格式：`absolute` 直接比较本版本，`delta` 比较 `candidate-baseline`。阈值仅是**示例规则，未经产品要求/实际基线校准**。Gate 不会因为缺少数据返回 Pass。

## 发布原则

- 提交代码、Schema、**合成**数据、测试、计算逻辑和运行说明。
- 不提交真实模型权重、完整测试/训练数据、原始私有 PDF、日志、trace、输出结果、真实 DPO Preference Pairs。
- 在 README 中把 `Implemented / Offline Tested / Pending Real Evaluation / Historical Reported` 分清楚；不要将合成样本成绩宣传成现网指标。
