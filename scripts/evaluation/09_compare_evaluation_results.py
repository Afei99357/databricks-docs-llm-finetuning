"""Rank checkpoint and RAG reference-score reports without another LLM call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-size", type=int, default=400)
    return parser.parse_args()


def rank_key(row: dict) -> tuple[float, float, int, float]:
    metrics = row["metrics"]
    return (
        -metrics["correctness_mean"],
        -metrics["completeness_mean"],
        metrics["unsupported_claims_total"],
        metrics["latency_ms_mean"],
    )


def main() -> None:
    options = args()
    run_dir = options.run_dir.resolve()
    scores_dir = run_dir / "evaluations" / "scores"
    reports = []
    for path in sorted(scores_dir.glob(f"*-benchmark_{options.benchmark_size}.json")):
        report = json.loads(path.read_text())
        reports.append({"candidate": report["candidate"], "metrics": report["metrics"], "report": str(path)})
    if not reports:
        raise FileNotFoundError(f"No {options.benchmark_size}-answer score reports under {scores_dir}")
    ranking = sorted(reports, key=rank_key)
    output = run_dir / "evaluations" / f"comparison_{options.benchmark_size}.json"
    output.write_text(json.dumps({"benchmark_size": options.benchmark_size, "ranking": ranking, "winner": ranking[0]}, indent=2) + "\n")
    print(json.dumps({"winner": ranking[0], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
