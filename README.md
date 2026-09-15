# Databricks documentation fine-tuning

This project creates a local documentation Q&A dataset, fine-tunes Qwen with
QLoRA, and evaluates the base model, adapters, checkpoints, and a local RAG
system on the same frozen benchmark.

The evaluation is designed to answer two questions:

- Does fine-tuning improve the base model for stable Databricks documentation?
- How do the best local model and RAG compare when the same independent judge
  scores both against the same reference answers?

## Requirements

- A local Databricks documentation SQLite corpus.
- An OpenAI-compatible Muse server for Q&A generation and the RAG application.
- An OpenAI-compatible Qwen 3.8 server as the independent evaluator.
- An AMD ROCm-capable GPU for QLoRA training and checkpoint answer generation.
- `uv` and `just`.

Create your local configuration and update the paths and servers as needed:

```bash
cp .env.example .env
just sync-rocm-deps
just check-rocm
```

`GENERATOR_*` configures Muse. `JUDGE_*` configures Qwen 3.8. Never use Muse
as the evaluator for RAG answers that Muse helped generate.

## Pipeline

```text
01–04  Build versioned training and frozen benchmark datasets
05     Fine-tune Qwen and retain the best checkpoint by validation loss
06     Generate base-model and checkpoint answers for the frozen benchmark
07     Generate RAG answers for that same benchmark
08     Score every saved answer source with Qwen 3.8
09     Rank the comparable score reports without another LLM call
```

Run from the repository root:

```bash
# Data
just page-aware-qa
just independent-qa
just paired-evaluation-qa
just export-dataset v001

# Training
just train configs/experiments/qwen35_9b_v001.yaml

# Evaluation. Run 06 and 07 sequentially: both use the local GPU stack.
just generate-checkpoint-answers artifacts/models/RUN_NAME artifacts/datasets/v001
just generate-rag-answers artifacts/models/RUN_NAME artifacts/datasets/v001

# Qwen scores all saved model and RAG answers. The final argument is workers.
just judge-all-answers artifacts/models/RUN_NAME 10
just compare-evaluation-results artifacts/models/RUN_NAME
```

Steps 06–08 are resumable. Generated answers are saved under
`artifacts/models/RUN_NAME/evaluations/answers/`; Qwen score reports are saved
under `evaluations/scores/`; step 09 writes `comparison_400.json`.

## Evaluation metrics

Qwen 3.8 scores each answer against its reference answer:

- `correctness_mean` — factual accuracy, averaged from 1 to 5.
- `completeness_mean` — coverage of required information, averaged from 1 to 5.
- `unsupported_claims_total` — unsupported factual statements across the set.
- `latency_ms_mean` — generation latency, when available.

Step 09 ranks by correctness, completeness, fewer unsupported claims, then
latency. This makes the RAG and checkpoint rows directly comparable.

## MLflow

Training records go to `databricks-docs-training`. Reference-scored evaluation
records go to `databricks-docs-answer-evaluation`. The latter contains the same
metric columns for base, checkpoint, adapter, and RAG rows.

Start the local UI with:

```bash
just mlflow-ui
```

Use `tags.smoke_test = 'false'` to hide smoke checks. The experiment names are
set in each YAML configuration with `training_experiment_name` and
`answer_evaluation_experiment_name`.

## Repository layout

```text
scripts/qa_generation/  Q&A generation and dataset export
scripts/training/       QLoRA training, probes, and MLflow helpers
scripts/evaluation/     Answer generation, scoring, and result ranking
configs/experiments/    Versioned experiment configurations
artifacts/              Local datasets, models, answers, and reports
```

Local artifacts, MLflow data, `.env`, and model environments are ignored by
Git. Do not change a YAML configuration after its run starts; copy it for each
material experiment change.
