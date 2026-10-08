# Integration and migration checklist

Current input: original `legacy/` source, user-held `data/` directory, existing fine-tuned BGE reranker and Qwen3-8B SFT checkpoints. Historical DPO checkpoint was lost.

1. **Protect originals.** Keep original project/code and model checkpoints as read-only snapshots. Do not commit `data/`, original PDFs, indexes, Mongo exports, real QA/Preference Pairs, or checkpoint weights to GitHub.
2. **Inspect actual data contracts.** Confirm `manual_text`, `manual_images`, Parent/Child metadata, unique IDs, `document_version` and source pages. The old uploaded Milvus/BM25 retrievers currently use `unique_id` but don't establish the target parent/version contract; the old `MilvusRetriever.retrieve_topk` is internally Dense+Sparse hybrid, not necessarily a strict independent Dense branch.
3. **Build canonical Parent/Image evidence.** Ensure every record has `evidence_key`, `evidence_group_id`, `kind`, `text_surrogate`, `knowledge_base_id`, `index_version`, `document_id`, `document_version`, `source_uri`, `source_pages`, `access_labels`, and `image_uri` for images. Persist Child -> Parent mapping and separate original image assets. Mongo is canonical; indices are derived and rebuildable.
4. **Create immutable Manifest.** For each `index_version`, record `document_versions` and validate Child↔Parent, Mongo↔index ID/content checks. Use `scripts/validate_evidence.py` for schema smoke, then implement stronger cross-index integrity checks before activation. Never silently infer version metadata.
5. **Replace adapter wiring.** Connect actual BM25 Child rank list, BGE-M3 Dense Child list, Chinese-CLIP image list to `RankedRetriever`. Expand all IDs through `MongoCanonicalStore.fetch_many`. The strict legacy adapter rejects old records missing metadata until migration.
6. **Validate reranker granularity.** Run saved tuned BGE checkpoint on Parent / Image text surrogate and check whether rankings/calibration hold. The new `HFSequenceClassificationScorer` uses raw single logits. Recalibrate the Router's coverage policy in that same raw score space.
7. **Prepare Router data.** Build labeled atomic-requirement x evidence matrices, form train/dev/held-out splits *before* negative generation/teacher labeling, then train a new 6-feature LR using `scripts/train_router.py`. Supply a dev-calibrated `--coverage-threshold`. Store the resulting JSON outside GitHub. Do not call this artifact the historical LR model.
8. **Replace the synthetic decomposer/rewriter.** Use real Qwen3-1.7B or another approved model with constrained outputs for atomic requirements and partial/failed query rewrites. Freeze requirements after Round1 for the final gate.
9. **Connect the DPO-first Answer Model.** The adapter and bridge are implemented in `src/controlled_answer/` and `src/controlled_rag/adapters/qwen3_dpo.py`. SFT is an **explicit baseline** using the saved checkpoint. Recover/retrain DPO before claiming the production target model is deployed. Use constrained generation when supported; it does not replace Pydantic/Citation Guard.
10. **Evaluate and release.** See `eval/README.md` for already implemented metric engines and synthetic offline tests. Run **real labeled** no-answer, injection, unauthorized-KB, fabricated citation, ambiguous queries, retrieval recall, reranker NDCG, classifier false-sufficient, round2 rescue, citation semantic support. Add request-level logs, bounded queue/deadline, staged index activation, regression and throughput tests. Only then describe the system as end-to-end operational.

## Frozen invariants to preserve

`Child Recall -> Parent Evidence -> Requirement Gate -> at most one Round2 -> FINAL Gate -> sealed final_evidence -> Qwen3-8B -> StructuredAnswer -> Citation Guard -> canonical metadata renderer`.

The real dataset `data/` was **not** supplied in this upload. Because its record contents are unknown, this package does not claim to have migrated them.
