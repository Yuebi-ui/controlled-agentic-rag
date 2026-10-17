"""Strict end-to-end, per-question evaluation for paired RAG experiments.

This module computes scores from *provided* labels; it does not label claims,
invoke a language model, or certify the origin of those labels.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

VARIANTS = ("basic_rag", "controlled_full")
STATES = {"answered", "abstained", "failed"}


def _required(row: dict, name: str):
    if name not in row:
        raise ValueError(f"missing field {name!r} in {row.get('id', '<unknown>')}")
    return row[name]


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not (0 <= value <= 1e9):
        raise ValueError(f"{name}: expected nonnegative finite number")
    return float(value)


def validate_rows(rows: Iterable[dict]) -> dict[str, dict[str, dict]]:
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    labels: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each JSONL record must be an object")
        uid = _required(row, "id")
        variant = _required(row, "variant")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("nonempty string ID required")
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError(f"invalid variant: {variant!r}")
        if uid in groups[variant]:
            raise ValueError(f"duplicate variant/query pair: {variant}/{uid}")
        gold = _required(row, "gold_answerable")
        if type(gold) is not bool:
            raise ValueError(f"{uid}: gold_answerable must be bool")
        if uid in labels and labels[uid] != gold:
            raise ValueError(f"{uid}: inconsistent gold_answerable across variants")
        labels[uid] = gold
        status = _required(row, "final_status")
        if status not in STATES:
            raise ValueError(f"{uid}: unknown final_status {status}")
        generated = _required(row, "generation_called")
        if type(generated) is not bool:
            raise ValueError(f"{uid}: generation_called must be bool")
        if status == "answered":
            if not generated:
                raise ValueError(f"{uid}: answered without generation")
            for field in ("answer_correct", "claim_supported", "citation_valid"):
                if type(_required(row, field)) is not bool:
                    raise ValueError(f"{uid}: {field} required as bool for answered cases")
            if variant == "controlled_full" and row.get("final_gate_state") != "sufficient":
                raise ValueError(f"{uid}: full pipeline answered without sufficient final gate")
        else:
            for field in ("answer_correct", "claim_supported", "citation_valid"):
                if _required(row, field) is not None:
                    raise ValueError(f"{uid}: {field} must be null when no answer was emitted")
        _number(_required(row, "latency_ms"), "latency_ms")
        n_rounds = _required(row, "retrieval_rounds")
        if type(n_rounds) is not int or n_rounds not in (1, 2):
            raise ValueError(f"{uid}: retrieval_rounds must be 1 or 2")
        if variant == "controlled_full" and status == "answered" and not row.get("sealed_final_present", False):
            raise ValueError(f"{uid}: answered without a sealed evidence set")
        groups[variant][uid] = row
    if not groups:
        raise ValueError("no records")
    first_name = sorted(groups)[0]
    fixed_ids = set(groups[first_name])
    for name, records in groups.items():
        if set(records) != fixed_ids:
            missing, extra = fixed_ids - set(records), set(records) - fixed_ids
            raise ValueError(f"{name}: unpaired sample IDs missing={len(missing)} extra={len(extra)}")
    if set(VARIANTS) - set(groups):
        raise ValueError("unpaired: both basic_rag and controlled_full are required")
    return dict(groups)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[min(len(sorted_values)-1, max(0, int((len(sorted_values)-1)*fraction + .99999999)))]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score(row: dict, *, strict: bool) -> bool:
    if not row["gold_answerable"]:
        return row["final_status"] == "abstained"
    if row["final_status"] != "answered" or not row["answer_correct"]:
        return False
    return not strict or (row["claim_supported"] and row["citation_valid"])


def summarize_end_to_end(rows: Iterable[dict], *, seed: int = 81,
                         resamples: int = 2000) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    data = validate_rows(rows)
    reports = {}
    for name, group in sorted(data.items()):
        cohort = list(group.values())
        answered = [r for r in cohort if r["final_status"] == "answered"]
        gold_answerable = [r for r in cohort if r["gold_answerable"]]
        gold_unanswerable = [r for r in cohort if not r["gold_answerable"]]
        latencies = sorted(r["latency_ms"] for r in cohort)
        reports[name] = {
            "n_all": len(cohort),
            "n_gold_answerable": len(gold_answerable),
            "n_gold_unanswerable": len(gold_unanswerable),
            "n_answered": len(answered),
            "n_failed": sum(r["final_status"] == "failed" for r in cohort),
            "overall_decision_correct": _mean([float(_score(r, strict=False)) for r in cohort]),
            "grounded_success_per_input": _mean([float(_score(r, strict=True)) for r in cohort]),
            "correct_answer_per_answerable": _mean(
                [float(r["final_status"] == "answered" and r["answer_correct"])
                 for r in gold_answerable]),
            "answer_faithfulness_given_answered": _mean(
                [float(r["claim_supported"]) for r in answered]),
            "citation_validity_given_answered": _mean(
                [float(r["citation_valid"]) for r in answered]),
            "false_refusal_per_answerable": _mean(
                [float(r["final_status"] == "abstained") for r in gold_answerable]),
            "unsafe_answer_per_unanswerable": _mean(
                [float(r["final_status"] == "answered") for r in gold_unanswerable]),
            "correct_refusal_per_unanswerable": _mean(
                [float(r["final_status"] == "abstained") for r in gold_unanswerable]),
            "round2_trigger_rate": _mean([float(r["retrieval_rounds"] == 2) for r in cohort]),
            "p95_latency_ms": _percentile(latencies, .95),
        }
        group_names = sorted(set(r.get("group", "unlabeled") for r in cohort))
        reports[name]["by_complexity"] = {
            label: {
                "n": sum(r.get("group", "unlabeled") == label for r in cohort),
                "grounded_success_per_input": _mean(
                    [float(_score(r, strict=True)) for r in cohort
                     if r.get("group", "unlabeled") == label]),
            }
            for label in group_names
        }
    first, second = data["basic_rag"], data["controlled_full"]
    ids = sorted(first)
    rng = random.Random(seed)
    paired = {}
    for key, strict in (("overall_decision_correct", False),
                        ("grounded_success_per_input", True)):
        diffs = [float(_score(second[uid], strict=strict)) -
                 float(_score(first[uid], strict=strict)) for uid in ids]
        boots = sorted(_mean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
        paired[key] = {
            "paired_n": len(ids),
            "delta_full_minus_basic": _mean(diffs),
            "bootstrap_ci_95": [_percentile(boots, .025), _percentile(boots, .975)],
            "seed": seed, "bootstrap_resamples": resamples,
        }
    return {"source_policy": "metrics computed from supplied records; origin independently auditable",
            "variants": reports, "paired": paired}


def summarize_outcome_ablation(rows: Iterable[dict], *, base: str = "controlled_full",
                               seed: int = 81, resamples: int = 2000) -> dict:
    """Score *provided* per-ID outcomes, not substitute a real configuration execution."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    by_variant = defaultdict(dict)
    for row in rows:
        uid, variant = _required(row, "id"), _required(row, "variant")
        success = _required(row, "grounded_success")
        if not isinstance(uid, str) or not isinstance(variant, str) or type(success) is not bool:
            raise ValueError("ablation requires string IDs/variants and boolean outcome")
        if uid in by_variant[variant]:
            raise ValueError("duplicate ablation variant/id")
        by_variant[variant][uid] = int(success)
    if base not in by_variant or len(by_variant) < 2:
        raise ValueError("missing baseline or ablation candidates")
    ids = set(by_variant[base])
    if not ids:
        raise ValueError("empty cohort")
    for name, item in by_variant.items():
        if set(item) != ids:
            raise ValueError(f"unpaired ablation for {name}")
    rng = random.Random(seed)
    variants = {}
    for name in sorted(by_variant):
        if name == base:
            continue
        diffs = [by_variant[base][uid] - by_variant[name][uid] for uid in sorted(ids)]
        bootstrap = sorted(_mean(rng.choices(diffs, k=len(diffs)))
                           for _ in range(resamples))
        variants[name] = {
            "paired_n": len(diffs),
            "variant_grounded_success": _mean(list(by_variant[name].values())),
            "full_grounded_success": _mean(list(by_variant[base].values())),
            "delta_full_minus_variant": _mean(diffs),
            "bootstrap_ci_95": [_percentile(bootstrap, .025), _percentile(bootstrap, .975)],
        }
    return {"base": base, "variants": variants, "seed": seed,
            "bootstrap_resamples": resamples}
