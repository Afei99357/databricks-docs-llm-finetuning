"""Run a manual prompt against a saved QLoRA checkpoint and log the probe to MLflow."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def training_experiment_name(config: dict) -> str:
    return str(config.get("training_experiment_name", config["experiment_name"]))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory created by 05_finetune_llm.py.")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the newest checkpoint in <run-dir>/checkpoints.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def newest_checkpoint(run_dir: Path) -> Path:
    candidates = [path for path in (run_dir / "checkpoints").glob("checkpoint-*") if (path / "adapter_config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No adapter checkpoints found under {run_dir / 'checkpoints'}")
    return max(candidates, key=lambda path: int(path.name.rsplit("-", 1)[1]))


def final_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    model_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    generation_config = model.base_model.config
    original_architectures = generation_config.architectures
    if original_architectures is None:
        generation_config.architectures = ["UnslothTextOnly"]
    try:
        output_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=None,
        )
    finally:
        generation_config.architectures = original_architectures
    answer = tokenizer.decode(output_ids[0, model_inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return answer.replace("<|im_end|>", "").split("\nuser\n", maxsplit=1)[0].strip()


def main() -> None:
    args = arguments()
    run_dir = args.run_dir.resolve()
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing {run_config_path}")
    run_config = json.loads(run_config_path.read_text())
    checkpoint = args.checkpoint.resolve() if args.checkpoint else newest_checkpoint(run_dir)
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json in {checkpoint}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint),
        max_seq_length=run_config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        text_only=True,
        device_map={"": 0},
        offload_embedding=False,
    )
    FastLanguageModel.for_inference(model)
    answer = final_answer(model, tokenizer, args.question, args.max_new_tokens)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "base_model": run_config["base_model"],
        "question": args.question,
        "answer": answer,
        "max_new_tokens": args.max_new_tokens,
    }
    output_dir = run_dir / "checkpoint_probes"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{checkpoint.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}")
    mlflow.set_experiment(training_experiment_name(run_config))
    with mlflow.start_run(run_name=f"probe-{checkpoint.name}-{run_config['run_name']}"):
        mlflow.set_tags({"run_kind": "checkpoint-probe", "parent_training_run": run_config["run_name"], "checkpoint": checkpoint.name, "smoke_test": str(bool(run_config.get("smoke_test", False))).lower()})
        mlflow.log_params({"base_model": run_config["base_model"], "max_new_tokens": args.max_new_tokens})
        mlflow.log_artifact(str(output_path), artifact_path="checkpoint_probes")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
