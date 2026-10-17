# Paired Pipeline ablation runner

Ablation is performed by **running each Pipeline variant on the same question IDs**, not by entering hypothetical scores into a CSV/JSONL.

##  Executable synthetic example

```bash
python eval/ablation/run_matrix.py --mode synthetic \
  --cases eval/data/agent_workflow_synthetic.jsonl \
  --matrix eval/data/ablation_matrix_synthetic.json \
  --output outputs/ablation_synthetic
```

The example rebuilds two Pipeline instances: `baseline` uses `max_retrieval_rounds=2`, and `without_round2` uses `max_retrieval_rounds=1`. That one-round variant does not generate when its final evidence is insufficient. `outputs/ablation_synthetic/` contains:

```text
ablation_cases.jsonl           # Matched per-question numeric outcomes
ablation_report.json           # Variant comparison and provenance
variants/baseline/workflow_cases.jsonl
variants/baseline/workflow_report.json
variants/without_round2/workflow_cases.jsonl
variants/without_round2/workflow_report.json
```

The demonstration may show differences because it is deterministic and designed to exercise Round2. It is **not** a real-world benefit estimate or an independent sample size sufficient for significance claims.

## Real model / retrieval variants

Provide a JSON matrix listing one baseline and at least one differently configured variant:

```json
{
  "baseline": "baseline",
  "variants": [
    {"name": "baseline", "settings": {"max_retrieval_rounds": 2}},
    {"name": "without_round2", "settings": {"max_retrieval_rounds": 1}}
  ]
}
```

 Register a project-local variant factory, with signature:

```python

def build(variant_name: str, settings: dict):
    # Build a NEW actual ControlledRAGPipeline with the requested components.
    # Return (pipeline, request_factory(case_id)).
    ...
```

Run:

```bash
python eval/ablation/run_matrix.py --mode real \
  --cases /private/frozen_workflow_cases.jsonl \
  --matrix /private/ablation_matrix.json \
  --factory my_project.ablation_factory:build \
  --output outputs/ablation_real
```

For BM25/Dense/CLIP/RRF, Parent-aware reranking, LR vs threshold routing and SFT/DPO ablations, the **factory must implement the real component switch**. This repository does not silently simulate those integrations. `max_retrieval_rounds` is checked against the requested setting; other custom settings remain the factory's responsibility. For a fair comparison, pin test IDs, corpus/index version, labels, prompts, model settings and evaluation rules.

### Metrics and caveats

The comparison computes matched-per-query mean differences and bootstrapped descriptive CIs (default seed 2026). Every metric has its own **paired metric sample count**; missing image/Parent Gold labels do not turn into zeros. Baseline and variant failures must remain represented as failed decisions, not be omitted to improve the score.

The recorded `parent_candidate_recall_at_10` measures the union of retrieval candidates across rounds, not a new reranker NDCG. `answerability_decision_correct` measures whether the pipeline answered or abstained according to corpus-level Gold, not semantic answer correctness. `answered_without_sufficient_gate` is a control invariant check.
