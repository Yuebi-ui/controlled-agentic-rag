# Experiment system

 **公开数据声明**：`RUN-EXAMPLE-2026-001` 的脱敏 JSONL 由项目维护者声明来自真实实验，来源标签为 `real_audited`。本 ZIP 可复算指标与验证文件 SHA-256，但缺少可独立核验的原始模型执行日志、Gold 标注与外部审计资料。

## 可用入口

- [300 Query 已发布记录与指标](published_runs/RUN-EXAMPLE-2026-001/RESULTS.md)
- [Manifest（SHA-256）](published_runs/RUN-EXAMPLE-2026-001/manifest.json)
- [机器可读指标目录](results_catalog.json)
- [离线评测与复算指引](../../eval/README.md)
- [评测 Schemas](../../eval/data/SCHEMAS.md) 与 [标注协议](../../eval/data/ANNOTATION_PROTOCOL.md)

## 研究模块

| 模块 | 当前文档 |
|---|---|
| Retrieval / Parent-aware Rerank | [retrieval.md](retrieval.md) |
| Evidence Routing | [routing.md](routing.md) |
| Round2 / Final Gate | [round2.md](round2.md) |
| Workflow / Ablation | [workflow_and_ablation.md](workflow_and_ablation.md) |
| Serving / Safety | [serving.md](serving.md) |

