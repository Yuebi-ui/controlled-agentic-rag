# Legacy — not the active Controlled Agentic RAG pipeline

This directory preserves the **original uploaded implementation**, including the old
LLM-based `EvidenceAgent`, the ORPO-related scripts, index builder and its previous
README, for source provenance and incremental migration.

It is deliberately **not** the primary package or the target serving entry point.
The old `src/pipeline/rag_pipeline.py` lacks the new final Round2 Evidence Gate,
and historical ORPO/DPO paths in old scripts must not be confused with the new
Qwen3-8B SFT baseline or a recovered DPO checkpoint.

The old implementation depends on private corpus/index paths, large models and
infrastructure that are not included. Running it requires a separate legacy setup.
