"""Generate resumable paired-benchmark answers for the base model and adapters."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

# Qwen3.5 currently exposes an incomplete ``architectures`` field. Disable
# Unsloth's optional fast-generation wrapper and retain Transformers generate.
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
from unsloth import FastLanguageModel


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


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


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_prefix(rows: list[dict], benchmark: list[dict], path: Path) -> None:
    if len(rows) > len(benchmark):
        raise ValueError(f"{path} contains more rows than the benchmark")
    for index, row in enumerate(rows):
        question = benchmark[index]["messages"][0]["content"]
        if row.get("benchmark_index") != index or row.get("question") != question:
            raise ValueError(f"{path} does not match the current benchmark at row {index}")


def generate_batch(model, tokenizer, questions: list[str]) -> tuple[list[str], int]:
    conversations = [[{"role": "user", "content": question}] for question in questions]
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    started = time.perf_counter()
    output = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 768,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    batch_latency_ms = round((time.perf_counter() - started) * 1000)
    prompt_length = inputs["input_ids"].shape[1]
    answers = [tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip() for row in output]
    return answers, batch_latency_ms


def main() -> None:
    options = args()
    if options.batch_size < 1:
        raise ValueError("batch size must be at least 1")
    run_dir, dataset_dir = options.run_dir.resolve(), options.dataset_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    benchmark = [json.loads(line) for line in (dataset_dir / "benchmark_400.jsonl").read_text().splitlines() if line.strip()][:options.limit]
    if not benchmark:
        raise ValueError("Benchmark is empty")

    answers_dir = run_dir / "evaluations" / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    for name, model_name in candidates(run_dir, config["base_model"]):
        output = answers_dir / f"{name}-benchmark_{len(benchmark)}.jsonl"
        partial = output.with_suffix(".jsonl.partial")
        if output.is_file():
            completed = load_rows(output)
            validate_prefix(completed, benchmark, output)
            if len(completed) == len(benchmark):
                print({"candidate": name, "status": "already-complete", "answers": len(completed)})
                continue
        rows = load_rows(partial)
        validate_prefix(rows, benchmark, partial)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
            text_only=True,
            device_map={"": 0},
            offload_embedding=False,
        )
        FastLanguageModel.for_inference(model)
        with partial.open("a") as handle, tqdm(total=len(benchmark), initial=len(rows), desc=f"{name} answers", unit="question") as progress:
            for start in range(len(rows), len(benchmark), options.batch_size):
                items = benchmark[start : start + options.batch_size]
                questions = [item["messages"][0]["content"] for item in items]
                answers, batch_latency_ms = generate_batch(model, tokenizer, questions)
                latency_ms = round(batch_latency_ms / len(items))
                for offset, (item, answer) in enumerate(zip(items, answers, strict=True)):
                    row = {
                        "candidate": name,
                        "benchmark_index": start + offset,
                        "question": item["messages"][0]["content"],
                        "reference": item["messages"][1]["content"],
                        "answer": answer,
                        "latency_ms": latency_ms,
                    }
                    handle.write(json.dumps(row) + "\n")
                    rows.append(row)
                handle.flush()
                progress.update(len(items))

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        partial.replace(output)
        print({"candidate": name, "status": "complete", "answers": len(rows)})


if __name__ == "__main__":
    main()
