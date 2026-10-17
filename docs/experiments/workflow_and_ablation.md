# Workflow, end-to-end quality and ablation


🔴 端到端联合成功的定义：Gold 可回答时必须 `answered && answer_correct && claim_supported && citation_valid`，Gold 不可回答时必须 `abstained`。失败、输出格式错误与错误引用不得自动从分母移除。报告分母按全体输入、Gold 可回答、已回答三个层次分别定义。


```bash
python eval/results/audit_published_run.py --root docs/experiments/published_runs/RUN-EXAMPLE-2026-001
python eval/ablation/run_matrix.py --mode synthetic --cases eval/data/agent_workflow_synthetic.jsonl --matrix eval/data/ablation_matrix_synthetic.json --output outputs/ablation_demo
```

