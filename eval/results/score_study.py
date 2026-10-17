"""Evaluate recorded paired outcomes from JSONL (source labels are caller-supplied)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controlled_rag.evaluation.e2e import summarize_end_to_end, summarize_outcome_ablation


def run(input_path: Path, output_path: Path, *, mode: str = "e2e",
        seed: int = 81, resamples: int = 2000) -> dict:
    raw = input_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if mode == "e2e":
        metrics = summarize_end_to_end(rows, seed=seed, resamples=resamples)
    elif mode == "ablation":
        metrics = summarize_outcome_ablation(rows, seed=seed, resamples=resamples)
    else:
        raise ValueError("mode must be e2e or ablation")
    result = {
        "mode": mode,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "note": "Input labels are assumed; scoring does not certify semantic or experimental origin.",
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["e2e", "ablation"], default="e2e")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    output = run(args.input, args.output, mode=args.mode, seed=args.seed,
                 resamples=args.resamples)
    print(json.dumps({"output": str(args.output), "row_count": output["row_count"],
                      "mode": output["mode"]}))


if __name__ == "__main__":
    main()
