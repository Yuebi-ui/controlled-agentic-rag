# Evaluation

`src/controlled_rag/evaluation/` 提供检索、Router、Round2、生成、Citation、安全、Workflow、Ablation 和 Benchmark 指标计算。`src/controlled_rag/evaluation/e2e.py` 提供有固定分母的配对评估。

## Offline tests & published scenario bundle

```bash
python -m pytest -q
python eval/run_all.py
python eval/results/audit_published_run.py --root docs/experiments/published_runs/RUN-EXAMPLE-2026-001
python eval/results/score_study.py --mode e2e --input docs/experiments/published_runs/RUN-EXAMPLE-2026-001/workflow_anonymized.jsonl --output outputs/e2e_scores.json --seed 113 --resamples 4000
```

