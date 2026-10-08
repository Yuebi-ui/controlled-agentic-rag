#!/usr/bin/env python3
"""Train and export the six-feature LR router from YOUR labeled data.

Separate train and dev splits BEFORE constructing labels or hard negatives.
The historical artifact is unavailable. This script does not recreate old metrics.

Input JSONL row:
  {"case_id":"train-1","label":"partial", "features": {
      "query_top1": 0.3, "query_top3_mean": 0.2, ...}}

Usage:
 python scripts/train_router.py --train path/to/train.jsonl \
   --dev path/to/dev.jsonl --coverage-threshold YOUR_CALIBRATED_SCORE \
   --out artifacts/router.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_rag.routing import FEATURE_NAMES, STATES


def read_rows(path: Path):
    x, y, ids = [], [], []
    with path.open(encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if set(row["features"]) != set(FEATURE_NAMES):
                raise ValueError(f"{path}:{n}: wrong feature keys")
            if row["label"] not in STATES:
                raise ValueError(f"{path}:{n}: unexpected label")
            if not row.get("case_id"):
                raise ValueError(f"{path}:{n}: case_id required for leakage check")
            x.append([float(row["features"][feature]) for feature in FEATURE_NAMES])
            y.append(row["label"])
            ids.append(row["case_id"])
    if not x or len(set(ids)) != len(ids):
        raise ValueError(f"empty/duplicate case IDs in {path}")
    return x, y, ids


def train(train_file: Path, dev_file: Path, threshold: float, output: Path):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train, train_ids = read_rows(train_file)
    x_dev, y_dev, dev_ids = read_rows(dev_file)
    if set(train_ids).intersection(dev_ids):
        raise ValueError("train/dev case_id overlap (data leakage)")
    if len(set(y_train)) != 3:
        raise ValueError("training split must include sufficient/partial/failed")
    if not np.isfinite(np.asarray(x_train)).all() or not np.isfinite(np.asarray(x_dev)).all():
        raise ValueError("non-finite feature")
    if not np.isfinite(threshold):
        raise ValueError("non-finite coverage threshold")
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipeline.fit(x_train, y_train)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    # Fold StandardScaler into LR so production inference needs only standard
    # library; verify logits match sklearn predictions on held-out DEV.
    coeff = classifier.coef_ / scaler.scale_[None, :]
    intercept = classifier.intercept_ - np.sum(
        classifier.coef_ * scaler.mean_[None, :] / scaler.scale_[None, :], axis=1)
    payload = {
        "artifact_type": "trained_multinomial_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "classes": list(classifier.classes_),
        "coefficients": coeff.tolist(), "intercepts": intercept.tolist(),
        "coverage_threshold": threshold,
        "note": "Newly trained artifact. Not the missing historical router weights.",
    }
    from controlled_rag.routing import LinearSoftmaxModel
    standalone = LinearSoftmaxModel(payload["classes"], payload["coefficients"], payload["intercepts"])
    predictions = [standalone.predict(tuple(row)) for row in x_dev]
    original = list(pipeline.predict(x_dev))
    if predictions != original:
        raise AssertionError("export/inference disagree; refusing to save router")
    accuracy = sum(p == y for p, y in zip(predictions, y_dev)) / len(y_dev)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved NEW router artifact to {output}; DEV accuracy={accuracy:.4f}, n={len(y_dev)}")
    print("Do NOT report DEV accuracy as held-out test or reproduce historical metrics.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--coverage-threshold", type=float, required=True,
                    help="calibrated on training/dev only, NOT a historical threshold")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    train(args.train, args.dev, args.coverage_threshold, args.out)


if __name__ == "__main__":
    main()
