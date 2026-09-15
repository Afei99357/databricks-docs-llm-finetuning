"""Compare one saved adapter benchmark result against RAG on the same questions."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Comparison(BaseModel):
    winner: str  # candidate, rag, or tie
    rationale: str


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--rag-url", default="http://127.0.0.1:8000/api/answer")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    model_results = json.loads(args.model_results.read_text())
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    judge = OpenAI(base_url=os.getenv("GENERATOR_BASE_URL"), api_key=os.getenv("GENERATOR_API_KEY", "local"))
    candidate = model_results["candidate"]
    output = run_dir / "evaluations" / f"{candidate}-vs-rag.json"
    partial = output.with_suffix(".jsonl.partial")
    if output.is_file():
        completed = json.loads(output.read_text())["rows"]
        if len(completed) == len(model_results["rows"]):
            print({"candidate": candidate, "status": "already-complete", "comparisons": len(completed)})
            return

    comparisons = load_jsonl(partial)
    if len(comparisons) > len(model_results["rows"]):
        raise ValueError(f"{partial} has more rows than {args.model_results}")
    for index, comparison in enumerate(comparisons):
        source = model_results["rows"][index]
        if comparison.get("question") != source["question"] or comparison.get("candidate") != source["answer"]:
            raise ValueError(f"{partial} does not match {args.model_results} at row {index}")

    with partial.open("a") as handle, tqdm(total=len(model_results["rows"]), initial=len(comparisons), desc=f"{candidate} vs RAG", unit="question") as progress:
        for row in model_results["rows"][len(comparisons) :]:
            request = urllib.request.Request(args.rag_url, data=json.dumps({"question": row["question"]}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                rag_answer = json.loads(response.read())["answer"]
            verdict = judge.chat.completions.parse(model=os.getenv("GENERATOR_MODEL"), messages=[{"role": "system", "content": "Choose candidate, rag, or tie based on correctness, completeness, and unsupported claims versus the reference. Return JSON only."}, {"role": "user", "content": f"QUESTION:\n{row['question']}\n\nREFERENCE:\n{row['reference']}\n\nCANDIDATE:\n{row['answer']}\n\nRAG:\n{rag_answer}"}], response_format=Comparison).choices[0].message.parsed
            comparison = {"question": row["question"], "reference": row["reference"], "candidate": row["answer"], "rag": rag_answer, "verdict": verdict.model_dump()}
            handle.write(json.dumps(comparison) + "\n")
            handle.flush()
            comparisons.append(comparison)
            progress.update(1)
    wins = {key: sum(item["verdict"]["winner"] == key for item in comparisons) for key in ("candidate", "rag", "tie")}
    output.write_text(json.dumps({"model_candidate": candidate, "wins": wins, "rows": comparisons}, indent=2) + "\n")
    partial.unlink()
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=f"{config['run_name']}-vs-rag"):
        mlflow.set_tags({"run_kind": "adapter-vs-rag", "parent_training_run": config["run_name"], "candidate": candidate})
        mlflow.log_metrics({f"{key}_wins": value for key, value in wins.items()})
        mlflow.log_artifact(str(output), artifact_path="rag_comparison")
    print(wins)


if __name__ == "__main__":
    main()
