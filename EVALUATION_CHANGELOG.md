# Evaluation upgrade — what changed from the two supplied ZIPs

This repository is a **full, merged update** of `Controlled-Agentic-RAG-GitHub.zip` plus `Qwen3-8B-DPO-Interface-Patch.zip`. The historical `legacy/` source is retained. No model checkpoint, real user corpus, Mongo backup, or real metric artifact is included.

## Merge / integration

- Integrated the standalone `controlled_answer` package and its DPO interface tests.
- Corrected package imports and editable installation discovery; DPO remains the default, SFT and Base are explicitly selected only for comparison.
- Added `controlled_rag.adapters.qwen3_dpo.PipelineQwen3AnswerModel` to translate between the standalone `GenerationRequest` interface and the main orchestrator's `(query, requirements, sealed Evidence)` interface.
- Fixed error classification: malformed DPO output or forged IDs now produce `MODEL_OUTPUT_OR_CITATION_INVALID`; unavailable DPO endpoint gives `GENERATOR_UNAVAILABLE`. Neither case falls back to SFT.
- Added opt-in Pipeline stage observer events for offline retrieval, reranking, routing, Round2 and citation tracing.

## New evaluation source

- `src/controlled_rag/evaluation/` — strict schemas, retrieval, reranker, router, round2, generation, citation, safety, benchmark, ablation, badcase, split-audit, release-gate, tracing and CLI.
- `eval/data/` — synthetic JSONL fixtures and gold-label protocols, **not measured model results**.
- `eval/run_all.py` — run metric engines and a synthetic hard-gate smoke test.
- `eval/generation/run_live.py` — future Base/SFT/DPO endpoint predictions for identically supplied sealed evidence; no automatic correctness/Faithfulness labels.
- `eval/safety/run_control_probes.py` — executable deterministic error-path probes. Prompt Injection still requires real-model attacker-outcome review.
- `eval/benchmark/locustfile.py`, `scripts/http_bench.py` — optional load generator and request-level trace to compute P95/QPS.
- `scripts/collect_eval_traces.py` — request-pinned metadata-only instrumentation; real backends require an explicitly provided factory.
- `.github/workflows/ci.yml` — lightweight contract tests, synthetic evaluations and smoke artifact collection.
- `eval/README.md`, `eval/data/SCHEMAS.md`, `eval/data/ANNOTATION_PROTOCOL.md`, `docs/EVALUATION.md`, README status updates.

## Explicit limits

**The code is implemented and testable offline; the full real-system metrics have NOT been reproduced.** No real DPO checkpoint was present, no Parent/Image manual Gold corpus was included, no real Qwen3 generation, vLLM serving, router calibration, GPU throughput, citation semantic support, or Prompt Injection resistance was verified. Synthetic fixtures show algorithms working, not real model KPI outcomes.
