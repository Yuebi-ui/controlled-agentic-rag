# Published experiment records — maintainer-reported

## Provenance

| 字段 |  |
|---|---|
| Run ID | `RUN-EXAMPLE-2026-001` |
| Query cohort | `300` |
| Gold answerable / unanswerable | `222 / 78` |
| Source type | `real_audited`（维护者声明，非第三方认证） |
| Real system verified | `false` |
| Independent human annotation | `false` |
| Physical GPU measurement | `false` |
| Paired bootstrap | `4,000` resamples, seeds `113 / 125` |

## End-to-end quality

| 指标 | Basic RAG | Controlled Agentic RAG |
|---|---:|---:|
|  Grounded success / input | 67.0% | **81.7%** |
|  Correct answer / answerable | 64.4% | **77.5%** |
|  Faithfulness / answered | 79.0% | **88.4%** |
|  Citation validity / answered | 97.0% | **100.0%** |
|  False refusal / answerable ↓ | 25.7% | **14.9%** |
|  Unanswerable false-answer ↓ | 6.4% | **1.3%** |
|  P95 E2E latency ↓ | 2.82s | **4.85s** |

Grounded Success paired Δ = **+14.7 pp**；95% bootstrap CI = **[+8.0, +21.0] pp**。

## Retrieval, reranker & evidence routing

| 指标 | Baseline | Proposed |
|---|---:|---:|
|  Child Recall@10 | 87.7% | **94.7%** |
|  Parent Recall@10 | 90.0% | **96.3%** |
|  Image Recall@10 | 58.3% | **88.3%** |
|  Reranker NDCG@5 | 0.801 | **0.851** |
|  Router Macro-F1 | 0.745 | **0.933** |
|  Router False-Sufficient ↓ | 14.2% | **4.1%** |
|  Round2 Rescue (eligible) | — | **71.3%** |
|  Final-gate violations | — | **0** |

## Base / SFT / DPO judgement set

| 指标 | Base | SFT | DPO |
|---|---:|---:|---:|
|  Schema validity | 94.0% | 98.7% | **100.0%** |
|  Correct / answerable | 69.4% | 80.2% | **86.9%** |
|  Faithfulness / answered | 80.3% | 89.8% | **95.7%** |
|  Citation support / answered | 73.4% | 87.9% | **94.7%** |
|  Unanswerable false-answer ↓ | 25.6% | 7.7% | **1.3%** |
|  Claim-support reviewer κ | 0.826 | 0.688 | **0.618** |


## Paired full-system ablation

| 配置 | Grounded success | Full − variant | 95% paired CI |
|---|---:|---:|---:|
|  `without_hybrid` | 69.3% | +12.3 pp | [+8.7, +16.0] pp |
|  `without_reranker` | 79.0% | +2.7 pp | [+0.7, +5.0] pp |
|  `without_router` | 70.7% | +11.0 pp | [+7.7, +14.7] pp |
|  `without_round2` | 66.3% | +15.3 pp | [+11.3, +19.7] pp |
|  `sft_instead_of_dpo` | 78.0% | +3.7 pp | [+1.7, +6.0] pp |

## Safety probe records

共 `300` 个攻击测试记录，总通过率 **95.7%**。🔴 未公开原始攻击负载或独立人工复核证据；`attack_succeeded` 的来源无法仅凭脱敏记录确认。

## Serving and capacity fields

> 🔴 `physical_hardware_measured=false`，Manifest 的设备标识仍为 `EXAMPLE-NOT-MEASURED`。这些字段可作统计复算，不能当作已经验证的物理 GPU 实测。

| Concurrency | N | Success | P95 | QPS | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
|  1 | 250 | 100.0% | 3.95s | 0.32 | 56.5 GB |
|  2 | 250 | 100.0% | 4.23s | 0.60 | 58.1 GB |
|  4 | 250 | 99.2% | 4.82s | 1.07 | 61.2 GB |
|  8 | 250 | 99.6% | 6.30s | 1.63 | 67.4 GB |
|  12 | 250 | 97.6% | 7.75s | 1.86 | 73.6 GB |


## Evidence chain

```text
cohort_anonymized.jsonl
      ├── retrieval_anonymized.jsonl ──────> Recall / NDCG
      ├── router_anonymized.jsonl ─────────> Macro-F1 / False-Sufficient
      ├── workflow_anonymized.jsonl ───────> E2E / Round2 / latency
      ├── generation_judgements.jsonl ─────> Base/SFT/DPO / reviewer agreement
      ├── ablation_outcomes.jsonl ─────────> paired ablation + bootstrap
      ├── safety_anonymized.jsonl ─────────> attack success rate
      └── benchmark_anonymized.jsonl ──────> P50/P95/P99 / QPS / resource fields
                         │
                         └── summary.json
                                  │
                                  └── manifest.json (SHA-256)
```

