"""Small dependency-free metric functions and explicit denominator policies."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

ROUTES = ('sufficient', 'partial', 'failed')


def divide(a: float, b: float) -> float | None:
    return a / b if b else None


def avg(nums: Sequence[float]) -> float | None:
    return divide(sum(nums), len(nums))


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return divide(2 * precision * recall, precision + recall) if precision + recall else 0.0


def at_k(ranked: Sequence[str], gold: set[str], k: int) -> dict[str, float]:
    if k < 1 or not gold:
        raise ValueError('k and gold IDs must be positive/nonempty')
    top = list(dict.fromkeys(ranked))[:k]
    relevant = [1 if x in gold else 0 for x in top]
    first = next((i for i, rel in enumerate(relevant, 1) if rel), None)
    dcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(relevant, 1))
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(len(gold), k) + 1))
    return {'recall': sum(relevant) / len(gold), 'hit': float(first is not None),
            'mrr': 1.0 / first if first else 0.0, 'ndcg': dcg / ideal}


def confusion_stats(labels: Sequence[str], preds: Sequence[str], classes=ROUTES) -> dict:
    if len(labels) != len(preds) or not labels:
        raise ValueError('nonempty paired labels required')
    if any(x not in classes for x in [*labels, *preds]):
        raise ValueError('unknown class label')
    counts = Counter(zip(labels, preds))
    class_rows: dict[str, dict] = {}
    macro_f1: list[float] = []
    for cls in classes:
        tp = counts[(cls, cls)]
        predicted = sum(counts[(a, cls)] for a in classes)
        actual = sum(counts[(cls, b)] for b in classes)
        precision, recall = divide(tp, predicted), divide(tp, actual)
        value = f1(precision, recall)
        # Convention: zero-support + zero-prediction classes have F1=0 (not excluded).
        if value is None:
            value = 0.0
        macro_f1.append(value)
        class_rows[cls] = {'support': actual, 'predicted': predicted,
                           'precision': precision, 'recall': recall, 'f1': value}
    return {'accuracy': sum(a == b for a, b in zip(labels, preds)) / len(labels),
            'macro_f1': avg(macro_f1), 'per_class': class_rows,
            'confusion': {a: {b: counts[(a, b)] for b in classes} for a in classes}}


def expected_calibration_error(probs: Sequence[Sequence[float]], gold: Sequence[str],
                               classes=ROUTES, bins: int = 10) -> float | None:
    """Top-label ECE, equal-width confidence bins and sample weighted."""
    if not probs:
        return None
    if len(probs) != len(gold) or bins <= 0:
        raise ValueError('probability and label lengths/bins invalid')
    counts = [0] * bins
    conf = [0.0] * bins
    acc = [0.0] * bins
    for row, label in zip(probs, gold):
        if len(row) != len(classes) or label not in classes:
            raise ValueError('class probability mismatch')
        certainty = max(row)
        idx = min(int(certainty * bins), bins - 1)
        winner = classes[row.index(certainty)]
        counts[idx] += 1
        conf[idx] += certainty
        acc[idx] += float(winner == label)
    n = len(probs)
    return sum(abs(conf[i] / counts[i] - acc[i] / counts[i]) * counts[i] / n
               for i in range(bins) if counts[i])


def multiclass_brier(probs: Sequence[Sequence[float]], gold: Sequence[str],
                     classes=ROUTES) -> float | None:
    """Unscaled multiclass Brier: mean(sum_c(p_c - 1[y=c])^2)."""
    if not probs:
        return None
    if len(probs) != len(gold):
        raise ValueError('unpaired probabilities')
    return avg([sum((p - float(label == cls)) ** 2 for p, cls in zip(row, classes))
                for row, label in zip(probs, gold)])


def nearest_rank(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    if percent < 0 or percent > 1:
        raise ValueError('invalid percentile')
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(percent * len(ordered)) - 1)])
