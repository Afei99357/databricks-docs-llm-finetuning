# MLflow experiment workflow

Use MLflow as the source of truth for training and evaluation lineage. Keep
human-authored configuration, code, and short reports in Git; keep large model
artifacts, raw outputs, and MLflow's SQLite store/artifacts out of Git.

## Before each training run

1. Copy the closest file in `configs/experiments/` to a new versioned file.
2. Create a dataset manifest with split counts and SHA-256 hashes.
3. Record the source corpus revision, curation rules, and category balance in
   the manifest before starting the run.
4. Never modify an experiment configuration or dataset manifest after its run
   begins.

## Run identity

Use names with model, dataset, material settings, and seed:

```text
qwen35-9b-sft-data-v003-lr1e-4-seed20260910
```

Each training run must log: base-model name and revision, dataset version and
hashes, hyperparameters, package/environment information, loss history,
hardware/time metrics, adapter path, tokenizer/chat-template configuration,
and final evaluation reports.

## Evaluation gate

Every candidate must use the fixed `adapters_n100_answer_only` sample set.
Promotion requires comparison against its matching base model, the current
best local adapter, and RAG. Record correctness, completeness, unsupported
claim rate, failures, citations, and latency. Treat the model as a candidate
until it beats its base without unacceptable regressions.

## Local commands

Create a dataset manifest:

```bash
The active step `scripts/qa_generation/04_export_final_datasets.py` now writes `dataset_manifest.json` automatically. This legacy command is retained only as historical reference.
```

Log the current 9B baseline without copying adapter weights:

```bash
uv run python scripts/log_existing_baseline.py \
  --training-dir artifacts/training/qwen3_5_9b_qlora \
  --evaluation-dir artifacts/evaluations/adapters_n100_answer_only
```

Open the UI:

```bash
just mlflow-ui
```
