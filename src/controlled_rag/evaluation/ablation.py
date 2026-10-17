"""Per-metric paired ablation comparisons; never compare unequal question cohorts."""
from __future__ import annotations

import random

from .metrics import avg, nearest_rank
from .schemas import AblationRecord


def evaluate_ablation(rows: list[AblationRecord], baseline: str = 'baseline',
                      bootstraps: int = 1000, seed: int = 2026) -> dict:
    if not rows:
        raise ValueError('empty ablation dataset')
    if bootstraps < 0:
        raise ValueError('bootstraps cannot be negative')
    by_variant: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        samples = by_variant.setdefault(row.variant, {})
        if row.id in samples:
            raise ValueError(f'duplicate ablation sample {row.id} in {row.variant}')
        samples[row.id] = row.metrics
    if baseline not in by_variant or len(by_variant) < 2:
        raise ValueError('baseline and at least one other ablation variant required')
    rng = random.Random(seed)
    report = {}
    for variant, examples in sorted(by_variant.items()):
        if variant == baseline:
            continue
        shared_ids = sorted(set(examples) & set(by_variant[baseline]))
        if not shared_ids:
            raise ValueError(f'no paired samples for variant {variant}')
        all_metrics = sorted(set().union(*(set(examples[uid]) & set(by_variant[baseline][uid])
                                           for uid in shared_ids)))
        if not all_metrics:
            raise ValueError(f'no common paired metric for variant {variant}')
        comparisons: dict[str, dict] = {}
        for metric in all_metrics:
            paired_ids = [uid for uid in shared_ids if metric in examples[uid]
                          and metric in by_variant[baseline][uid]]
            diffs = [examples[uid][metric] - by_variant[baseline][uid][metric]
                     for uid in paired_ids]
            if not diffs:
                continue
            mean = avg(diffs)
            if bootstraps > 0:
                means = [sum(rng.choices(diffs, k=len(diffs))) / len(diffs)
                         for _ in range(bootstraps)]
                ci = [nearest_rank(means, .025), nearest_rank(means, .975)]
            else:
                ci = None
            comparisons[metric] = {
                'paired_metric_sample_count': len(paired_ids),
                'omitted_shared_cases_missing_metric': len(shared_ids) - len(paired_ids),
                'baseline_mean': avg([by_variant[baseline][uid][metric] for uid in paired_ids]),
                'variant_mean': avg([examples[uid][metric] for uid in paired_ids]),
                'paired_mean_delta': mean,
                'bootstrap_95pct_ci': ci,
            }
        report[variant] = {
            'paired_sample_count': len(shared_ids),
            'dropped_unpaired_baseline_cases': len(by_variant[baseline]) - len(shared_ids),
            'dropped_unpaired_variant_cases': len(examples) - len(shared_ids),
            'metrics': comparisons,
        }
    return {
        'baseline': baseline,
        'variant_comparisons': report,
        'bootstrap_resamples': bootstraps,
        'seed': seed,
        'note': ('Per-metric paired question IDs; bootstrap descriptive only. '
                 'Synthetic data are not production performance evidence.'),
    }
