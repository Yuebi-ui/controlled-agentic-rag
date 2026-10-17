"""Metrics for an anonymized published-run evidence bundle.

These functions only recompute metrics from supplied records. They never certify
that records came from real models, humans, indexes, or hardware.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable


LABELS = ("sufficient", "partial", "failed")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def percentile(values, q: float):
    values = sorted(values)
    if not values:
        return None
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[idx]


def recall_at_k(gold, ranked, k=10):
    gold = set(gold)
    if not gold:
        return None
    return len(gold.intersection(ranked[:k])) / len(gold)


def ndcg_binary(relevance, k=5):
    rel = list(relevance[:k])
    dcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def classification_metrics(gold, pred, labels=LABELS):
    gold, pred = list(gold), list(pred)
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold and pred must be same nonzero length")
    f1s = []
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        if g not in labels or p not in labels:
            raise ValueError("unknown label")
        confusion[g][p] += 1
    for label in labels:
        tp = confusion[label][label]
        predicted = sum(confusion[g][label] for g in labels)
        actual = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)
    correct = sum(confusion[l][l] for l in labels)
    insufficient = sum(sum(confusion[g].values()) for g in ("partial", "failed"))
    false_sufficient = sum(confusion[g]["sufficient"] for g in ("partial", "failed"))
    return {
        "n": len(gold),
        "accuracy": correct / len(gold),
        "macro_f1": sum(f1s) / len(f1s),
        "false_sufficient_rate": false_sufficient / insufficient if insufficient else None,
        "confusion": confusion,
    }


def summarize_retrieval(rows: Iterable[dict]):
    rows = list(rows)
    child_bm25, child_hybrid, parent_bm25, parent_hybrid = [], [], [], []
    image_bm25, image_hybrid = [], []
    before, after = [], []
    for r in rows:
        child_bm25.append(recall_at_k(r["gold_child_ids"], r["bm25_child_ranked_ids"]))
        child_hybrid.append(recall_at_k(r["gold_child_ids"], r["hybrid_child_ranked_ids"]))
        parent_bm25.append(recall_at_k(r["gold_parent_ids"], r["bm25_parent_ranked_ids"]))
        parent_hybrid.append(recall_at_k(r["gold_parent_ids"], r["hybrid_parent_ranked_ids"]))
        if r["gold_image_ids"]:
            image_bm25.append(recall_at_k(r["gold_image_ids"], r["bm25_image_ranked_ids"]))
            image_hybrid.append(recall_at_k(r["gold_image_ids"], r["hybrid_image_ranked_ids"]))
        before.append(ndcg_binary(r["reranker_before_relevance"]))
        after.append(ndcg_binary(r["reranker_after_relevance"]))
    return {
        "n_queries": len(rows),
        "child_recall_at_10": {"bm25": mean(child_bm25), "hybrid": mean(child_hybrid)},
        "parent_recall_at_10": {"bm25": mean(parent_bm25), "hybrid": mean(parent_hybrid)},
        "image_recall_at_10": {"n": len(image_bm25), "bm25": mean(image_bm25),
                               "hybrid": mean(image_hybrid)},
        "reranker_ndcg_at_5": {"before": mean(before), "after": mean(after)},
    }


def summarize_router(rows: Iterable[dict]):
    rows = list(rows)
    gold = [r["gold_state"] for r in rows]
    return {
        "n_queries": len(rows),
        "rule": classification_metrics(gold, [r["rule_predicted_state"] for r in rows]),
        "learned": classification_metrics(gold, [r["learned_predicted_state"] for r in rows]),
    }


def summarize_round2(workflow_rows: Iterable[dict]):
    rows = [r for r in workflow_rows if r["variant"] == "controlled_full"]
    eligible = [r for r in rows if r.get("round2_eligible", False)]
    triggered = [r for r in rows if r["retrieval_rounds"] == 2]
    rescued = [r for r in eligible if r.get("round2_rescued", False)]
    violations = [r for r in rows
                  if r["final_status"] == "answered"
                  and r["final_gate_state"] != "sufficient"]
    return {
        "n_queries": len(rows),
        "triggered_count": len(triggered),
        "trigger_rate": len(triggered) / len(rows) if rows else None,
        "eligible_answerable_triggered_count": len(eligible),
        "rescued_count": len(rescued),
        "rescue_rate_eligible": len(rescued) / len(eligible) if eligible else None,
        "final_gate_violation_count": len(violations),
    }


def _binary_f1(gold, pred, positive=True):
    tp = sum(g == positive and p == positive for g, p in zip(gold, pred))
    fp = sum(g != positive and p == positive for g, p in zip(gold, pred))
    fn = sum(g == positive and p != positive for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def cohens_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    labels = set(x for x, _ in pairs) | set(y for _, y in pairs)
    po = mean(x == y for x, y in pairs)
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[l] / len(pairs)) * (cb[l] / len(pairs)) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def summarize_generation(rows: Iterable[dict]):
    rows = list(rows)
    report = {}
    for model in sorted(set(r["model_variant"] for r in rows)):
        cohort = [r for r in rows if r["model_variant"] == model]
        answered = [r for r in cohort if r["predicted_status"] == "answered"]
        answerable = [r for r in cohort if r["gold_answerable"]]
        unanswerable = [r for r in cohort if not r["gold_answerable"]]
        report[model] = {
            "n": len(cohort),
            "schema_validity": mean(r["schema_valid"] for r in cohort),
            "correctness_per_answerable": mean(
                r["predicted_status"] == "answered" and r["adjudicated"]["answer_correct"]
                for r in answerable),
            "faithfulness_given_answered": mean(
                r["adjudicated"]["claim_supported"] for r in answered),
            "citation_support_given_answered": mean(
                r["adjudicated"]["citation_supported"] for r in answered),
            "unsafe_answer_per_unanswerable": mean(
                r["predicted_status"] == "answered" for r in unanswerable),
            "abstention_f1": _binary_f1(
                [not r["gold_answerable"] for r in cohort],
                [r["predicted_status"] == "abstained" for r in cohort]),
            "claim_support_reviewer_kappa": cohens_kappa(
                [r["reviewer_a"]["claim_supported"] for r in answered],
                [r["reviewer_b"]["claim_supported"] for r in answered]),
        }
    return report


def summarize_safety(rows: Iterable[dict]):
    rows = list(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["attack_category"]].append(r)
    return {
        "n": len(rows),
        "pass_rate": mean(not r["attack_succeeded"] for r in rows),
        "by_attack": {
            k: {"n": len(v), "pass_rate": mean(not r["attack_succeeded"] for r in v)}
            for k, v in sorted(groups.items())
        },
    }


def summarize_benchmark(rows: Iterable[dict]):
    rows = list(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[int(r["concurrency"])].append(r)
    result = {}
    for c, group in sorted(groups.items()):
        successes = [r for r in group if r["success"]]
        window_ms = max(r["end_ms"] for r in group) - min(r["start_ms"] for r in group)
        result[str(c)] = {
            "n": len(group),
            "success_rate": mean(r["success"] for r in group),
            "p50_latency_ms": percentile([r["latency_ms"] for r in successes], .50),
            "p95_latency_ms": percentile([r["latency_ms"] for r in successes], .95),
            "p99_latency_ms": percentile([r["latency_ms"] for r in successes], .99),
            "mean_ttft_ms": mean(r["ttft_ms"] for r in successes),
            "mean_tpot_ms": mean(r["tpot_ms"] for r in successes),
            "qps": len(successes) / (window_ms / 1000) if window_ms > 0 else None,
            "peak_vram_gb": max(r["vram_gb"] for r in group),
            "mean_gpu_util_pct": mean(r["gpu_util_pct"] for r in group),
        }
    return result
