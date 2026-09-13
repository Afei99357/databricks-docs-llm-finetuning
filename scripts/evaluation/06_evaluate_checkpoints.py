"""Generate and judge paired-benchmark answers for a base model and saved adapters."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Judgement(BaseModel):
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    unsupported_claims: int = Field(ge=0)
    rationale: str


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=400)
    return parser.parse_args()


def answer(model, tokenizer, question: str) -> str:
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=768, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def candidates(run_dir: Path, base_model: str) -> list[tuple[str, str]]:
    values = [("base", base_model)]
    for name in ("best_adapter", "adapter"):
        path = run_dir / name
        if (path / "adapter_config.json").is_file():
            values.append((name, str(path)))
    for path in sorted((run_dir / "checkpoints").glob("checkpoint-*")):
        if (path / "adapter_config.json").is_file():
            values.append((path.name, str(path)))
    return values


def main() -> None:
    options = args()
    run_dir, dataset_dir = options.run_dir.resolve(), options.dataset_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    benchmark = [json.loads(line) for line in (dataset_dir / "benchmark_400.jsonl").read_text().splitlines() if line.strip()][:options.limit]
    if not benchmark:
        raise ValueError("Benchmark is empty")
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    judge = OpenAI(base_url=os.getenv("GENERATOR_BASE_URL"), api_key=os.getenv("GENERATOR_API_KEY", "local"))
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment(config["experiment_name"])
    output_dir = run_dir / "evaluations"
    output_dir.mkdir(exist_ok=True)
    for name, model_name in candidates(run_dir, config["base_model"]):
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=config["max_seq_length"], dtype=None, load_in_4bit=True, text_only=True, device_map={"": 0}, offload_embedding=False)
        FastLanguageModel.for_inference(model)
        rows, totals = [], {"correctness": 0, "completeness": 0, "unsupported_claims": 0}
        for item in benchmark:
            question, reference = item["messages"][0]["content"], item["messages"][1]["content"]
            started = time.perf_counter(); generated = answer(model, tokenizer, question); latency_ms = round((time.perf_counter() - started) * 1000)
            judged = judge.chat.completions.parse(model=os.getenv("GENERATOR_MODEL"), messages=[{"role": "system", "content": "Score the candidate answer against the reference. Return JSON only. Correctness and completeness are 1-5. Count unsupported factual claims."}, {"role": "user", "content": f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nCANDIDATE:\n{generated}"}], response_format=Judgement).choices[0].message.parsed
            totals["correctness"] += judged.correctness; totals["completeness"] += judged.completeness; totals["unsupported_claims"] += judged.unsupported_claims
            rows.append({"question": question, "reference": reference, "answer": generated, "latency_ms": latency_ms, "judge": judged.model_dump()})
        del model
        metrics = {"correctness_mean": totals["correctness"] / len(rows), "completeness_mean": totals["completeness"] / len(rows), "unsupported_claims_total": totals["unsupported_claims"], "latency_ms_mean": sum(row["latency_ms"] for row in rows) / len(rows)}
        output = output_dir / f"{name}-benchmark_{len(rows)}.json"
        output.write_text(json.dumps({"candidate": name, "metrics": metrics, "rows": rows}, indent=2) + "\n")
        with mlflow.start_run(run_name=f"{config['run_name']}-benchmark-{name}"):
            mlflow.set_tags({"run_kind": "paired-benchmark", "parent_training_run": config["run_name"], "candidate": name})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output), artifact_path="benchmark")
        print({"candidate": name, **metrics})


if __name__ == "__main__":
    main()
