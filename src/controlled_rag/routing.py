"""Atomic requirement features and reproducible multinomial LR inference.

Trained LR weights and the calibrated coverage threshold are externally supplied.
The production router requires an explicitly trained artifact; no fake weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import Evidence, RouteState

FEATURE_NAMES = (
    "query_top1", "query_top3_mean", "requirement_max_mean",
    "requirement_max_min", "coverage_ratio", "missing_requirement_count",
)
STATES: tuple[RouteState, ...] = ("sufficient", "partial", "failed")


class Scorer(Protocol):
    def score_many(self, query: str, evidences: Sequence[Evidence]) -> list[float]: ...


class Decomposer(Protocol):
    def decompose(self, original_query: str) -> Sequence[str]: ...


class Rewriter(Protocol):
    def rewrite_missing(self, query: str, missing_requirements: Sequence[str]) -> str: ...
    def rewrite_failed(self, query: str) -> str: ...


@dataclass(frozen=True)
class RouterDecision:
    state: RouteState
    features: tuple[float, float, float, float, float, float]
    missing_requirements: tuple[str, ...]
    covered_best_keys: tuple[str, ...]
    raw_model_state: RouteState | None = None


class LabelModel(Protocol):
    def predict(self, features: tuple[float, ...]) -> RouteState: ...


class LinearSoftmaxModel:
    """Pure-Python inference of a **trained** sklearn multinomial LR export."""

    def __init__(self, classes: list[str], coefficients: list[list[float]], intercepts: list[float]):
        if set(classes) != set(STATES) or len(classes) != 3 or len(coefficients) != 3 or len(intercepts) != 3:
            raise ValueError("three LR classes required: sufficient, partial, failed")
        if any(len(coef) != len(FEATURE_NAMES) for coef in coefficients):
            raise ValueError("LR coefficients must match six-feature contract")
        if not all(math.isfinite(v) for row in coefficients for v in row):
            raise ValueError("non-finite LR coefficient")
        if not all(math.isfinite(x) for x in intercepts):
            raise ValueError("non-finite LR intercept")
        self.classes = classes
        self.coefficients = coefficients
        self.intercepts = intercepts

    def predict(self, features: tuple[float, ...]) -> RouteState:
        if len(features) != 6 or not all(math.isfinite(x) for x in features):
            raise ValueError("six finite LR features required")
        logits = [bias + sum(w * x for w, x in zip(row, features))
                  for row, bias in zip(self.coefficients, self.intercepts)]
        return self.classes[max(range(3), key=lambda i: logits[i])]  # type: ignore[return-value]

    @classmethod
    def from_json(cls, path: str | Path) -> tuple["LinearSoftmaxModel", float]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("trained router feature order mismatch")
        threshold = payload["coverage_threshold"]
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("coverage threshold missing or invalid")
        return cls(payload["classes"], payload["coefficients"], payload["intercepts"]), float(threshold)


class EvidenceRouter:
    def __init__(self, scorer: Scorer, model: LabelModel, coverage_threshold: float):
        if not math.isfinite(coverage_threshold):
            raise ValueError("coverage threshold must be finite")
        self.scorer = scorer
        self.model = model
        self.threshold = coverage_threshold

    def route(self, requirements: Sequence[str], query_scores: Sequence[float],
              evidences: Sequence[Evidence]) -> RouterDecision:
        if not requirements or any(not req.strip() for req in requirements):
            raise ValueError("original atomic requirements must be nonempty")
        if len(query_scores) != len(evidences) or not all(math.isfinite(s) for s in query_scores):
            raise ValueError("scores/evidence mismatch or non-finite scores")
        row_best: list[float] = []
        missing: list[str] = []
        best_keys: list[str] = []
        for requirement in requirements:
            scores = self.scorer.score_many(requirement, evidences) if evidences else []
            if len(scores) != len(evidences) or not all(math.isfinite(s) for s in scores):
                raise ValueError("requirement-evidence score matrix is invalid")
            if not scores:
                # No evidence: define zero features, but always fail closed below.
                row_best.append(0.0)
                missing.append(requirement)
            else:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                best = scores[best_idx]
                row_best.append(best)
                if best >= self.threshold:
                    best_keys.append(evidences[best_idx].evidence_key)
                else:
                    missing.append(requirement)
        ordered_query = sorted(query_scores, reverse=True)
        top1 = ordered_query[0] if ordered_query else 0.0
        top3 = sum(ordered_query[:3]) / len(ordered_query[:3]) if ordered_query else 0.0
        n = len(requirements)
        features = (
            float(top1), float(top3), float(sum(row_best) / n), float(min(row_best)),
            float((n - len(missing)) / n), float(len(missing)),
        )
        raw_state = self.model.predict(features) if evidences else "failed"
        # Respect the trained 3-way label, but impose a safety floor: a model
        # cannot say Sufficient when ANY original atomic requirement is missing.
        # A Partial label with nothing covered is also treated as Failed.
        if raw_state == "sufficient" and missing:
            state: RouteState = "partial" if len(missing) < n else "failed"
        elif raw_state == "partial" and len(missing) == n:
            state = "failed"
        elif raw_state == "partial" and not missing:
            state = "failed"  # conflicting classifier/coverage signal; no generation
        else:
            state = raw_state
        return RouterDecision(state, features, tuple(missing),
                              tuple(dict.fromkeys(best_keys)), raw_state)
