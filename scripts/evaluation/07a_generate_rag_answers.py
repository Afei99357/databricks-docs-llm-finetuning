"""Generate resumable local-RAG answers for a saved checkpoint benchmark report."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--rag-url", default="http://127.0.0.1:8000/api/answer")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rag_answer(row: dict, rag_url: str, retries: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                rag_url,
                data=json.dumps({"question": row["question"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                answer = json.loads(response.read())["answer"]
            return {
                "benchmark_index": row["benchmark_index"],
                "question": row["question"],
                "reference": row["reference"],
                "candidate": row["answer"],
                "rag": answer,
            }
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    raise RuntimeError(
        f"RAG could not answer benchmark index {row['benchmark_index']} after {retries} attempts: {last_error}"
    ) from last_error


def main() -> None:
    options = args()
    if options.workers < 1 or options.retries < 1:
        raise ValueError("workers and retries must each be at least 1")
    run_dir = options.run_dir.resolve()
    model_results = json.loads(options.model_results.resolve().read_text())
    candidate = model_results["candidate"]
    source_rows = [
        {**row, "benchmark_index": index}
        for index, row in enumerate(model_results["rows"])
    ]
    output = run_dir / "evaluations" / f"{candidate}-rag-answers_{len(source_rows)}.jsonl"
    partial = output.with_suffix(".jsonl.partial")
    if output.is_file():
        completed = load_jsonl(output)
        if len(completed) == len(source_rows):
            print({"candidate": candidate, "status": "already-complete", "rag_answers": len(completed)})
            return

    rows = load_jsonl(partial)
    completed_indexes = {row["benchmark_index"] for row in rows}
    source_indexes = {row["benchmark_index"] for row in source_rows}
    if len(completed_indexes) != len(rows) or not completed_indexes.issubset(source_indexes):
        raise ValueError(f"{partial} does not match {options.model_results}")
    pending = [row for row in source_rows if row["benchmark_index"] not in completed_indexes]

    with partial.open("a") as handle, ThreadPoolExecutor(max_workers=options.workers) as pool, tqdm(total=len(source_rows), initial=len(rows), desc=f"{candidate} RAG answers", unit="question") as progress:
        futures = {
            pool.submit(rag_answer, row, options.rag_url, options.retries): row["benchmark_index"]
            for row in pending
        }
        failures = []
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                failures.append((futures[future], str(error)))
                progress.write(f"RAG index {futures[future]} failed: {error}")
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            progress.update(1)

    if failures:
        details = "; ".join(f"{index}: {error}" for index, error in failures[:3])
        raise RuntimeError(
            f"{candidate}: {len(failures)} RAG answer(s) still failed after retries ({details}). "
            "Completed answers were saved; rerun the command to retry only the unresolved questions."
        )
    rows.sort(key=lambda row: row["benchmark_index"])
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    partial.unlink()
    print({"candidate": candidate, "status": "complete", "rag_answers": len(rows), "output": str(output)})


if __name__ == "__main__":
    main()
