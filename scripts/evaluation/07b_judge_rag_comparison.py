"""Judge saved checkpoint and RAG answers against the shared reference answers."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENTS = threading.local()


class Comparison(BaseModel):
    winner: str  # candidate, rag, or tie
    rationale: str


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--rag-results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judge_settings() -> tuple[str, str]:
    return (
        os.getenv("JUDGE_BASE_URL", os.environ["GENERATOR_BASE_URL"]),
        os.getenv("JUDGE_MODEL", os.environ["GENERATOR_MODEL"]),
    )


def judge_client() -> OpenAI:
    base_url, _ = judge_settings()
    if not hasattr(CLIENTS, "judge"):
        CLIENTS.judge = OpenAI(base_url=base_url, api_key=os.getenv("JUDGE_API_KEY", os.getenv("GENERATOR_API_KEY", "local")))
    return CLIENTS.judge


def judge_comparison(row: dict, max_tokens: int, retries: int) -> dict:
    _, model = judge_settings()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            verdict = judge_client().chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Choose candidate, rag, or tie based on correctness, completeness, and unsupported claims versus the reference. Return exactly one compact JSON object with winner and a rationale of at most two sentences.",
                    },
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{row['question']}\n\nREFERENCE:\n{row['reference']}\n\nCANDIDATE:\n{row['candidate']}\n\nRAG:\n{row['rag']}",
                    },
                ],
                response_format=Comparison,
                max_tokens=max_tokens,
                temperature=0,
                reasoning_effort="none",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=90,
            ).choices[0].message.parsed
            return {**row, "verdict": verdict.model_dump()}
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    raise RuntimeError(
        f"Judge could not compare benchmark index {row['benchmark_index']} after {retries} attempts: {last_error}"
    ) from last_error


def main() -> None:
    options = args()
    if options.workers < 1 or options.max_tokens < 1 or options.retries < 1:
        raise ValueError("workers, max tokens, and retries must each be at least 1")
    run_dir = options.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    model_results = json.loads(options.model_results.resolve().read_text())
    candidate = model_results["candidate"]
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    judge_base_url, judge_model = judge_settings()
    try:
        OpenAI(base_url=judge_base_url, api_key=os.getenv("JUDGE_API_KEY", os.getenv("GENERATOR_API_KEY", "local"))).models.list()
    except Exception as error:
        raise RuntimeError(f"Cannot reach the judge at {judge_base_url}. Start the judge server, then retry.") from error

    rag_rows = load_jsonl(options.rag_results.resolve())
    model_rows = model_results["rows"]
    if len(rag_rows) != len(model_rows):
        raise ValueError("RAG and candidate result files have different row counts")
    for index, (rag, model) in enumerate(zip(rag_rows, model_rows, strict=True)):
        if rag.get("benchmark_index") != index or rag.get("question") != model["question"] or rag.get("candidate") != model["answer"]:
            raise ValueError(f"RAG results do not match candidate results at row {index}")

    output = run_dir / "evaluations" / f"{candidate}-vs-rag.json"
    partial = output.with_suffix(".jsonl.partial")
    if output.is_file():
        completed = json.loads(output.read_text())["rows"]
        if len(completed) == len(rag_rows):
            print({"candidate": candidate, "status": "already-complete", "comparisons": len(completed)})
            return
    comparisons = load_jsonl(partial)
    completed_indexes = {row["benchmark_index"] for row in comparisons}
    if len(completed_indexes) != len(comparisons) or not completed_indexes.issubset({row["benchmark_index"] for row in rag_rows}):
        raise ValueError(f"{partial} does not match {options.rag_results}")
    pending = [row for row in rag_rows if row["benchmark_index"] not in completed_indexes]

    with partial.open("a") as handle, ThreadPoolExecutor(max_workers=options.workers) as pool, tqdm(total=len(rag_rows), initial=len(comparisons), desc=f"{candidate} vs RAG judging", unit="question") as progress:
        futures = {
            pool.submit(judge_comparison, row, options.max_tokens, options.retries): row["benchmark_index"]
            for row in pending
        }
        failures = []
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                failures.append((futures[future], str(error)))
                progress.write(f"Judge index {futures[future]} failed: {error}")
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            comparisons.append(row)
            progress.update(1)

    if failures:
        details = "; ".join(f"{index}: {error}" for index, error in failures[:3])
        raise RuntimeError(
            f"{candidate}: {len(failures)} comparison(s) still failed after retries ({details}). "
            "Completed verdicts were saved; rerun the command to retry only the unresolved questions."
        )
    comparisons.sort(key=lambda row: row["benchmark_index"])
    wins = {key: sum(item["verdict"]["winner"] == key for item in comparisons) for key in ("candidate", "rag", "tie")}
    output.write_text(json.dumps({"model_candidate": candidate, "judge_model": judge_model, "wins": wins, "rows": comparisons}, indent=2) + "\n")
    partial.unlink()
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=f"{config['run_name']}-vs-rag"):
        mlflow.set_tags({"run_kind": "adapter-vs-rag", "parent_training_run": config["run_name"], "candidate": candidate, "judge_model": judge_model})
        mlflow.log_metrics({f"{key}_wins": value for key, value in wins.items()})
        mlflow.log_artifact(str(output), artifact_path="rag_comparison")
    print({"candidate": candidate, "judge_model": judge_model, **wins})


if __name__ == "__main__":
    main()
