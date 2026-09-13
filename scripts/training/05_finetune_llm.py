"""Fine-tune a compatible Qwen model with QLoRA and mandatory MLflow tracking.

The script trains only on the immutable ``training.jsonl`` export. Paired
benchmark data is deliberately never loaded here; it is used later to compare
the base model and saved checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.mlflow_tracking import MLflowTrainerCallback, fail_run, finish_run, load_experiment_config, start_run


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Use the deterministic small run from the config.")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run", action="store_true", help="Required before GPU training can begin.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_messages(row: dict) -> bool:
    messages = row.get("messages", [])
    return len(messages) == 2 and messages[0].get("role") == "user" and messages[1].get("role") == "assistant" and bool(messages[0].get("content", "").strip()) and bool(messages[1].get("content", "").strip())


def prompt_completion(row: dict) -> dict:
    return {"prompt": [row["messages"][0]], "completion": [row["messages"][1]]}


def main() -> None:
    args = arguments()
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    dataset_dir = (PROJECT_ROOT / config["dataset_dir"]).resolve()
    training_path = dataset_dir / "training.jsonl"
    selection_path = dataset_dir / "benchmark_selection_50.jsonl"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not training_path.is_file() or not selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run scripts/qa_generation/04_export_final_datasets.py before fine-tuning.")
    manifest = json.loads(manifest_path.read_text())
    expected_hash = manifest["training.jsonl"]["sha256"]
    if sha256(training_path) != expected_hash:
        raise RuntimeError("training.jsonl does not match its immutable dataset manifest.")
    if sha256(selection_path) != manifest["benchmark_selection_50.jsonl"]["sha256"]:
        raise RuntimeError("benchmark_selection_50.jsonl does not match its immutable dataset manifest.")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("A ROCm-enabled PyTorch GPU is required. Use utils/rocm-run to launch this script.")

    raw = load_dataset("json", data_files={"train": str(training_path)}, split="train")
    dataset = raw.filter(valid_messages, desc="validating training messages")
    if len(dataset) != len(raw):
        raise ValueError(f"Rejected {len(raw) - len(dataset)} malformed training examples")
    if args.smoke_test:
        dataset = dataset.shuffle(seed=config["seed"]).select(range(min(config["smoke_test_examples"], len(dataset))))
    dataset = dataset.map(prompt_completion, remove_columns=dataset.column_names, desc="preparing prompt/completion data")
    selection = load_dataset("json", data_files={"selection": str(selection_path)}, split="selection")
    selection = selection.filter(valid_messages, desc="validating checkpoint-selection messages")
    selection = selection.map(prompt_completion, remove_columns=selection.column_names, desc="preparing checkpoint-selection data")

    run_dir = PROJECT_ROOT / "artifacts" / "models" / config["run_name"]
    checkpoint_dir = run_dir / "checkpoints"
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "config_path": str(config_path), "training_path": str(training_path), "selection_path": str(selection_path), "smoke_test": args.smoke_test, "examples": len(dataset), "selection_examples": len(selection)}
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (run_dir / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0)}, indent=2) + "\n")
    print({"model": config["base_model"], "examples": len(dataset), "run_dir": str(run_dir), "smoke_test": args.smoke_test})
    if not args.run:
        print("Validated data and configuration. Add --run when ready to start GPU training.")
        return

    start_run(PROJECT_ROOT, config_path, manifest_path)
    try:
        mlflow.log_artifact(str(run_dir / "environment.json"), artifact_path="environment")
        model, tokenizer = FastVisionModel.from_pretrained(model_name=config["base_model"], max_seq_length=config["max_seq_length"], dtype=None, load_in_4bit=True, use_gradient_checkpointing="unsloth", text_only=True)
        lora = config["lora"]
        model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none", random_state=config["seed"], use_rslora=False, loftq_config=None)
        interval = config["smoke_test_save_steps"] if args.smoke_test else config["save_steps"]
        eval_interval = config["smoke_test_eval_steps"] if args.smoke_test else config["eval_steps"]
        if interval != eval_interval:
            raise ValueError("save and evaluation intervals must match when selecting the best checkpoint")
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, eval_dataset=selection, callbacks=[MLflowTrainerCallback()], args=SFTConfig(output_dir=str(checkpoint_dir), max_length=config["max_seq_length"], per_device_train_batch_size=config["per_device_train_batch_size"], per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"], num_train_epochs=config["num_train_epochs"], max_steps=config["smoke_test_max_steps"] if args.smoke_test else -1, warmup_ratio=config["warmup_ratio"], lr_scheduler_type=config["lr_scheduler_type"], optim="adamw_8bit", bf16=True, fp16=False, gradient_checkpointing=True, logging_steps=config["logging_steps"], eval_strategy="steps", eval_steps=eval_interval, save_strategy="steps", save_steps=interval, save_total_limit=config["save_total_limit"], load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, report_to="none", seed=config["seed"], completion_only_loss=True, packing=False))
        result = trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
        trainer.save_state()
        if not trainer.state.best_model_checkpoint:
            raise RuntimeError("No best checkpoint was selected; increase training steps beyond the evaluation interval.")
        best_checkpoint = Path(trainer.state.best_model_checkpoint)
        best_checkpoint_dir = run_dir / "best_checkpoint"
        shutil.copytree(best_checkpoint, best_checkpoint_dir, dirs_exist_ok=True)
        model.save_pretrained(adapter_dir)
        best_adapter_dir = run_dir / "best_adapter"
        model.save_pretrained(best_adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(best_adapter_dir)
        metrics = {**result.metrics, "train_examples": len(dataset), "selection_examples": len(selection), "best_checkpoint": str(best_checkpoint), "best_eval_loss": trainer.state.best_metric}
        (run_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        mlflow.set_tags({"best_checkpoint": str(best_checkpoint), "best_adapter": str(best_adapter_dir)})
        mlflow.log_artifacts(str(best_checkpoint_dir), artifact_path="best_checkpoint")
        mlflow.log_artifacts(str(best_adapter_dir), artifact_path="best_adapter")
        finish_run(run_dir, metrics, log_adapter=True)
    except Exception:
        fail_run()
        raise


if __name__ == "__main__":
    main()
