"""Judge saved checkpoint answers with the local Muse llama.cpp server."""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENTS = threading.local()


class Judgement(BaseModel):
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    unsupported_claims: int = Field(ge=0)
    rationale: str


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judge_client() -> OpenAI:
    if not hasattr(CLIENTS, "judge"):
        CLIENTS.judge = OpenAI(
            base_url=os.environ["GENERATOR_BASE_URL"],
            api_key=os.getenv("GENERATOR_API_KEY", "local"),
        )
    return CLIENTS.judge


def judge_answer(row: dict) -> dict:
    verdict = judge_client().chat.completions.parse(
        model=os.environ["GENERATOR_MODEL"],
        messages=[
            {
                "role": "system",
                "content": "Score the candidate answer against the reference. Return JSON only. Correctness and completeness are 1-5. Count unsupported factual claims.",
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{row['question']}\n\nREFERENCE:\n{row['reference']}\n\nCANDIDATE:\n{row['answer']}",
            },
        ],
        response_format=Judgement,
    ).choices[0].message.parsed
    return {**row, "judge": verdict.model_dump()}


def metrics(rows: list[dict]) -> dict[str, float | int]:
    return {
        "correctness_mean": sum(row["judge"]["correctness"] for row in rows) / len(rows),
        "completeness_mean": sum(row["judge"]["completeness"] for row in rows) / len(rows),
        "unsupported_claims_total": sum(row["judge"]["unsupported_claims"] for row in rows),
        "latency_ms_mean": sum(row["latency_ms"] for row in rows) / len(rows),
    }


def main() -> None:
    options = args()
    if options.workers < 1:
        raise ValueError("workers must be at least 1")
    run_dir = options.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    try:
        OpenAI(
            base_url=os.environ["GENERATOR_BASE_URL"],
            api_key=os.getenv("GENERATOR_API_KEY", "local"),
        ).models.list()
    except Exception as error:
        raise RuntimeError(
            f"Cannot reach the Muse judge at {os.environ['GENERATOR_BASE_URL']}. Start the llama.cpp server, then retry."
        ) from error
    answers_dir = run_dir / "evaluations" / "answers"
    answer_files = sorted(answers_dir.glob("*-benchmark_*.jsonl"))
    if not answer_files:
        raise ValueError(f"No generated answer files found in {answers_dir}")

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment(config["experiment_name"])
    for answers_path in answer_files:
        answers = load_jsonl(answers_path)
        if not answers:
            raise ValueError(f"{answers_path} is empty")
        candidate = answers[0]["candidate"]
        if any(row.get("candidate") != candidate for row in answers):
            raise ValueError(f"{answers_path} contains more than one candidate")
        output = run_dir / "evaluations" / f"{candidate}-benchmark_{len(answers)}.json"
        partial = output.with_suffix(".jsonl.partial")
        if output.is_file():
            completed = json.loads(output.read_text())["rows"]
            if len(completed) == len(answers):
                print({"candidate": candidate, "status": "already-complete", **metrics(completed)})
                continue

        judged_rows = load_jsonl(partial)
        completed_indexes = {row["benchmark_index"] for row in judged_rows}
        if len(completed_indexes) != len(judged_rows) or not completed_indexes.issubset({row["benchmark_index"] for row in answers}):
            raise ValueError(f"{partial} does not match {answers_path}")
        pending = [row for row in answers if row["benchmark_index"] not in completed_indexes]

        with partial.open("a") as handle, ThreadPoolExecutor(max_workers=options.workers) as pool, tqdm(total=len(answers), initial=len(judged_rows), desc=f"{candidate} judging", unit="answer") as progress:
            futures = {pool.submit(judge_answer, row): row["benchmark_index"] for row in pending}
            for future in as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                judged_rows.append(row)
                progress.update(1)

        judged_rows.sort(key=lambda row: row["benchmark_index"])
        candidate_metrics = metrics(judged_rows)
        output.write_text(json.dumps({"candidate": candidate, "metrics": candidate_metrics, "rows": judged_rows}, indent=2) + "\n")
        partial.unlink()
        with mlflow.start_run(run_name=f"{config['run_name']}-benchmark-{candidate}"):
            mlflow.set_tags({"run_kind": "paired-benchmark", "parent_training_run": config["run_name"], "candidate": candidate})
            mlflow.log_metrics(candidate_metrics)
            mlflow.log_artifact(str(output), artifact_path="benchmark")
        print({"candidate": candidate, **candidate_metrics})


if __name__ == "__main__":
    main()
