# Controlled Agentic RAG — Architecture / Trade-offs

## Overview

The primary state machine is `ControlledRAGPipeline.run`. All decisions depend on the **original user query**, its fixed atomic requirements, and canonical, request-pinned evidence. The LLM is a generator, **not** an authority on whether evidence is sufficient or on what source metadata exists.

## Data flow and trust boundaries

| Stage | Input | Output | Deterministic invariant |
|---|---|---|---|
| Resolve/Pin | request principal, KB/index pointer | `RequestContext`, `IndexManifest` | One request, one `index_version` |
| Retrieve | query, three rank lists | `RetrievalHit` | Hits carry IDs and version, **not trusted content** |
| RRF | sparse/dense/visual ranks | fused hits | Unweighted `1/(60+rank)`, exact ID dedup |
| Parent Collapse | Child hits | unique Parent/Image keys | Sibling Child hits do not exhaust Candidate Budget |
| Canonical Expansion | keys + request | `Evidence` | One batch fetch, permission, source/version checks |
| Rerank | original query, Parent/Image text surrogate | ranked evidence | Fine-tuned BGE checkpoint must be validated on this input distribution |
| Evidence Routing | atomic requirements x evidence support scores | 6 features, LR class | False-Sufficient is critical; classifier may not override missing requirements |
| Round2 | Partial / Failed | merged candidate pool | One optional Round2, keep R1 covered evidence, bounded candidates |
| Final Gate | original query and atomic requirements | Sufficient / Insufficient | Cannot generate on insufficient evidence |
| Sealing | final ranked/validated evidence | scoped E1/E2/... IDs | Only sealed IDs can be cited |
| Generation | query, requirements, sealed evidence | JSON text | No arbitrary retrieval or free-text citation metadata |
| Guard | Pydantic `StructuredAnswer` + sealed Evidence | validated blocks | No forged ID, cross-version, missing metadata or unauthorized citations |
| Renderer | validated IDs, canonical metadata | answer + citations | Pages/paths are from canonical store, never from model |

## Routing contract

The six frozen features are: (1) Query Top1, (2) Query Top3 Mean, (3) Requirement Max Mean, (4) Requirement Max Min, (5) Coverage Ratio, (6) Missing Requirement Count. A 3-class Logistic Regression consumes these in a fixed order. The actual training coefficients and final coverage threshold were not uploaded and **must not be invented**. `scripts/train_router.py` trains/exports a *new* model from appropriately separated labeled data.

Round1 may be Sufficient, Partial, or Failed. Partial rewrites **only missing requirements**. Failed rewrites **the overall question**. Both go through the same bounded Round2 retrieval unit. The last Rerank and Gate always use the **original** query and requirements.

## Error model

- `abstained`: semantically insufficient evidence or explicit model refusal (not a server failure).
- `failed`: inconsistent/stale/unauthorized evidence, invalid generation structure, fabricated citations, or insufficient available retrieval branches.
- `answered`: route allowed generation, output passed structure and deterministic citation validation.

The deterministic guard **does not establish semantic entailment**; Claim-level Citation Semantic Support is a separate model/human evaluation metric.

## Demo vs production

Synthetic DemoScorer, DemoLabelModel, DemoDecomposer and DemoGenerator are teaching/test doubles. They do **not** stand in for BGE-M3, Qwen3-1.7B, Qwen3-8B, or the trained LR classifier in any real benchmark. Heavy legacy source code is retained in `legacy/` for controlled migration; it is not imported into the new runtime.

Deferred integration: genuine three independent BM25/BGE-M3 Dense/Chinese-CLIP indices, batched versioned canonical Mongo store, real authorization service, calibrated LR artifact, decomposer/rewriter models, Qwen3-8B DPO as target, SFT/Base as explicit baselines with structured output, distributed deployment admission control, real index lifecycle rollout (the repository only has in-process policy helpers), production observability, regression, and capacity benchmark.
