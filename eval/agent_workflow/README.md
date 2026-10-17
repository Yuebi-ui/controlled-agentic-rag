# Agent workflow benchmark

Evaluate **execution decisions** in Controlled Agentic RAG, rather than using a generic tool-use Agent benchmark that does not reflect this system's evidence control flow.

## Run with deterministic synthetic components

```bash
python eval/agent_workflow/run.py --mode synthetic \
  --cases eval/data/agent_workflow_synthetic.jsonl \
  --output outputs/workflow_synthetic
```

The runner executes Pipeline requests (not prewritten metric rows) and writes `workflow_cases.jsonl` and `workflow_report.json` under the selected output directory. The shipped case labels are invented for contract testing only. Outputs are gitignored by default.

## JSONL case contract

Each line contains:

```json
{"id":"q001","query":"...","gold_answerable":true,"gold_round1_state":"partial","expected_round2":true,"gold_parent_ids":["p1"],"gold_image_ids":[]}
```

- `gold_round1_state`: independent annotation of what first-round evidence supports, not the Router's predicted label;
- `expected_round2`: independent policy label for the normal two-round workflow (`gold_round1_state != sufficient`);
- `gold_answerable`: whether the complete authorized corpus could answer, *not* whether the current retrieved candidates suffice;
- `gold_parent_ids` and `gold_image_ids`: optional source relevance labels; no labels means corresponding Recall metrics are unavailable.

Router labels must be reviewed again if an ablation changes the initial candidate set, because route sufficiency depends on *observed evidence*, not only on the question. No unknown keys, empty IDs or duplicate case IDs are accepted.

## Run with a real Pipeline

```bash
python eval/agent_workflow/run.py --mode real \
  --cases /private/frozen_workflow_cases.jsonl \
  --factory my_project.eval_factory:build \
  --output outputs/workflow_real
```

The factory must return `(pipeline: ControlledRAGPipeline, request_factory: callable(case_id) -> RequestContext)`. It must connect actual retrievers, calibrated Router, canonical store, generator, and permissions. There is **no** automatic substitution of synthetic models when a real factory fails or is absent.

`workflow_report.json` reports classification counts, trigger precision/recall, decision accuracy, eligible rescue and fail-closed violations. For any nontrivial claims, additionally measure answer correctness, faithfulness and citation semantic support with independent external annotations using their own suites. The runner stores per-case **metadata**, not raw answer text; nevertheless keep logs private until URI/access metadata are reviewed.
