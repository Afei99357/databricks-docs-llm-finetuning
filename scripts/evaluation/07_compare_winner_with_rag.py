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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Comparison(BaseModel):
    winner: str  # candidate, rag, or tie
    rationale: str


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
    comparisons = []
    for row in model_results["rows"]:
        request = urllib.request.Request(args.rag_url, data=json.dumps({"question": row["question"]}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            rag_answer = json.loads(response.read())["answer"]
        verdict = judge.chat.completions.parse(model=os.getenv("GENERATOR_MODEL"), messages=[{"role": "system", "content": "Choose candidate, rag, or tie based on correctness, completeness, and unsupported claims versus the reference. Return JSON only."}, {"role": "user", "content": f"QUESTION:\n{row['question']}\n\nREFERENCE:\n{row['reference']}\n\nCANDIDATE:\n{row['answer']}\n\nRAG:\n{rag_answer}"}], response_format=Comparison).choices[0].message.parsed
        comparisons.append({"question": row["question"], "reference": row["reference"], "candidate": row["answer"], "rag": rag_answer, "verdict": verdict.model_dump()})
    wins = {key: sum(item["verdict"]["winner"] == key for item in comparisons) for key in ("candidate", "rag", "tie")}
    output = run_dir / "evaluations" / f"{model_results['candidate']}-vs-rag.json"
    output.write_text(json.dumps({"model_candidate": model_results["candidate"], "wins": wins, "rows": comparisons}, indent=2) + "\n")
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=f"{config['run_name']}-vs-rag"):
        mlflow.set_tags({"run_kind": "adapter-vs-rag", "parent_training_run": config["run_name"], "candidate": model_results["candidate"]})
        mlflow.log_metrics({f"{key}_wins": value for key, value in wins.items()})
        mlflow.log_artifact(str(output), artifact_path="rag_comparison")
    print(wins)


if __name__ == "__main__":
    main()
