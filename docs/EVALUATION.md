# Evaluation implementation status

- **Implemented & offline tested:** JSONL strict contracts, Parent/Image retrieval and reranker metrics, router multiclass + safety/calibration, Round2 coverage/retention, generation Base/SFT/DPO scorecards, citation metadata + independent semantic annotation support, attack-outcome scorecards, latency/QPS aggregation, paired ablation, deterministic bad-case regression, split-leak audit, release gate and CI.
- **Executed on synthetic fixtures only:** `eval/run_all.py` (includes simulated model labels); `eval/safety/run_control_probes.py` (tests *deterministic* guards); metadata-only `scripts/collect_eval_traces.py --mode synthetic` (uses DemoGenerator). No pretrained models or real PDFs required.
- **Pending real evidence:** Parent/Image Gold relevance, tuned Reranker, router label training/calibration, real Round2 rescue, real Base/SFT/DPO generation, independently reviewed claim/citation support, model Prompt Injection robustness, GPU-backed load tests and historical metric reproduction.
- **Not automatically scored:** Semantic truth, hallucinations, Prompt Injection success, and claim entailment. Model-generated citations and Pydantic alone cannot validate these.

See the full [Evaluation & Reproducibility guide](../eval/README.md), and [schemas/denominators](../eval/data/SCHEMAS.md). Code can support future full experiments; the repository must not present the artificial demonstration numbers as Tesla production results.
