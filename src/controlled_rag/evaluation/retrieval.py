"""Parent and image retrieval metrics, per-query macro averages."""
from __future__ import annotations

from .metrics import at_k, avg
from .schemas import RetrievalRecord, RerankerRecord


def evaluate_retrieval(rows: list[RetrievalRecord], ks=(1, 5, 10)) -> dict:
    if not rows:
        raise ValueError('empty retrieval dataset')
    output = {'sample_count': len(rows), 'parent_labeled_count': 0, 'image_labeled_count': 0}
    for category in ('parent', 'image'):
        subset = [r for r in rows if getattr(r, f'gold_{category}_ids')]
        output[f'{category}_labeled_count'] = len(subset)
        for k in ks:
            for metric in ('recall', 'hit', 'mrr', 'ndcg'):
                vals = [at_k(getattr(r, f'ranked_{category}_ids'), set(getattr(r, f'gold_{category}_ids')), k)[metric]
                        for r in subset]
                output[f'{category}_{metric}_at_{k}'] = avg(vals)
    return output


def evaluate_reranker(rows: list[RerankerRecord], ks=(1, 5, 10)) -> dict:
    if not rows:
        raise ValueError('empty reranker dataset')
    result: dict = {'sample_count': len(rows)}
    for k in ks:
        for metric in ('recall', 'mrr', 'ndcg'):
            for stage in ('before', 'after'):
                result[f'{stage}_{metric}_at_{k}'] = avg([
                    at_k(getattr(row, stage), set(row.positive_ids), k)[metric]
                    for row in rows])
            result[f'delta_{metric}_at_{k}'] = (result[f'after_{metric}_at_{k}'] -
                                               result[f'before_{metric}_at_{k}'])
    pairs_correct = 0.0
    pairs_total = 0
    for row in rows:
        if row.after_scores is None:
            continue
        positives = set(row.positive_ids) & set(row.after)
        negatives = set(row.after) - set(row.positive_ids)
        for positive in positives:
            for negative in negatives:
                a, b = row.after_scores[positive], row.after_scores[negative]
                pairs_total += 1
                pairs_correct += 1.0 if a > b else 0.5 if a == b else 0.0
    result['pair_count'] = pairs_total
    result['pairwise_accuracy'] = pairs_correct / pairs_total if pairs_total else None
    return result
