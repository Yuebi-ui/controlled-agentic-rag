# Implementation status and evidence ledger

## A. Code confirmed in this repository

The new `src/controlled_rag` package includes an in-process validated manifest-pointer policy (`index_registry.py`) and a bounded local admission mechanism (`admission.py`), both tested with synthetic inputs but not wired to the real production environment.

The new `src/controlled_rag` package implements typed request/index/evidence contracts, unweighted RRF, exact-ID dedup, Parent Collapse, batch canonical lookup, permission/version checks, requirement-evidence six-feature extraction, 3-class trained LR JSON inference, explicit maximum two retrieval rounds, a final evidence gate, scoped evidence IDs, StructuredAnswer validation, deterministic citation checks and source metadata rendering.

Test scope is **synthetic** and mostly uses in-memory fakes. CI runs the no-GPU tests. A passing test means these mechanics behave correctly for the tested inputs. It does **not** verify Tesla retrieval/LLM quality, end-to-end throughput, calibration or deployed security controls.

## B. Adapters present but unverified with real data

- `adapters/models.py`: local Hugging Face sequence-classification checkpoint scorer. Need to verify actual BGE weights and their Parent/Image input format and calibration.
- `src/controlled_answer/` + `adapters/qwen3_dpo.py`: DPO-first (target), SFT and Base (explicit baselines), with real Pipeline protocol bridge, tests using simulated vLLM responses. DPO checkpoint is absent; real endpoint compatibility is unverified.
- `adapters/storage.py`: new Mongo canonical collection schema, **not** automatically compatible with legacy `manual_text` records.
- `adapters/index_hits.py`: strict legacy Document to new Hit bridge. It deliberately fails if old indices lack `parent_id`, `index_version` or `knowledge_base_id`; no automatic assertion that old index is compatible.
- `scripts/train_router.py`: infrastructure to train/export a **new** LR artifact from properly labeled data. The original trained LR weights and final threshold are absent.

## C. Not implemented / not reproduced

Real PDF Parent/Child rebuild and OCR/image text surrogates; fully migrated three independent retrieval backends; Qwen3-1.7B decomposer/rewrite; full auth service and principal identity integration; cache/production distributed index pointer lifecycle; Redis, distributed backpressure and request-wide deadline controls; real DPO data + training; executed real model quality/safety/integration benchmarks; deployed production release gates. The corresponding evaluation **metric code**, fixture smoke checks, metadata trace hooks and offline gates are now implemented under `src/controlled_rag/evaluation/` and `eval/` but they are not real-model evidence. The legacy parser/retrievers and ORPO scripts are archived, not a production-ready replacement.

## D. Metrics provenance: reported vs verified

The uploaded interview/architecture documents contain historical records such as Child Recall@10 92.2%, Parent-aware NDCG@5 0.868, False-Sufficient 4.4%, Round2 Rescue 70.8%, Citation Validity 100%, Citation Semantic Support 95.1%, and C4 P95 5.24 s. These values are **documented historical claims**, not results produced by this new GitHub code or verified against original raw artifacts in this delivery. Do not badge the new branch as having reproduced them.

In an interview, distinguish **historical reported metric**, **architecture now implemented in source**, **offline test passing**, and **planned real-world integration**. The lost DPO weights do not block current code work; DPO metrics are not demonstrated here.

## E. Newly implemented evaluation code (not historical reproduction)

`eval/README.md` documents ten strict JSONL suites and release checks; `eval/run_all.py` runs synthetic, non-model smoke checks. `scripts/collect_eval_traces.py` captures actual new Pipeline stage IDs/statuses from **synthetic** scenarios; no automatic semantic labels are invented. Tests and CI protect formula corner cases and explicit DPO adapter error classifications. Historical metrics listed above remain unverified.
